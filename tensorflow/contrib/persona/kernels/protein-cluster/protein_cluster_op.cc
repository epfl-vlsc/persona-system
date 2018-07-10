
#include <cstdint>
#include <pthread.h>
#include <queue>
#include <chrono>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/params.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/candidate_map.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/protein_cluster.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_environment.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_executor.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;
  using namespace format;

  // The AGDProteinClusterOp operates as a node in a ring architecture
  // All nodes in the ring are fed by a central input queue, which feeds chunks of proteins 
  // from multiple datasets (genomes)
  //
  // chunks are pre-encoded into string type tensors by AGDChunkToTensorOp, so 
  // that TF can seamlessly pipe chunks across the network when the ring spans 
  // multiple servers. 
  //
  // Each node is also connected to its neighbors via neighbor_queues (forming the ring)
  // When a node Compute()'s, it will prefer to dequeue a chunk from its neighbor (advancing the ring), if the
  // neighbor queue is empty it will attempt to dequeue from the input queue. If the input queue 
  // is closed and empty, it will dequeue and wait on the neighbor queue.
  // 
  // A sequence number tracks the position of each chunk in the ring. 
  // The sequence for a chunk is set to 0 if dequeued from the primary input queue.
  // The node then "owns" this chunk and will own the clusters produced by proteins
  // that were not previously added to any clusters. 
  //
  // If the sequence number for a dequeued chunk is equal to the ring size,
  // then the chunk has been seen and evaluated by all nodes in the ring. 
  // Any non-added proteins are used to seed new clusters owned by this node. 
  //
  // Otherwise, the dequeued chunk proteins are evaluated against the 
  // clusters owned by this node, and marked 'added' if they are added to any clusters.
  //
  // If this chunk was the last chunk, this node is done. The maintained cluster matches 
  // (i.e. the alignments between proteins in each cluster) are encoded into tensors
  // and enqueued so that a downstream AGDClusterAggregateOp can aggregate matches from
  // all clusters in the ring. 
  //
  // The ring itself and all the queues and dataflow is constructed via the 
  // TF python API. See persona-shell/modules/protein_cluster/*

  char* Denormalise(const char* str, int len) {
  auto ret = new char[(len + 1) * sizeof(char)];
  int i;

  for (i = 0; i < len; ++i) {
    ret[i] = 'A' + str[i];
  }

  ret[len] = '\0';

  return ret;
}
  
  class AGDProteinClusterOp : public OpKernel {
  public:
    AGDProteinClusterOp(OpKernelConstruction *context) : OpKernel(context) {
      // ring ID is compared to sequence ID of incoming chunks to see if we have processed it before
      // indicating it has made a full trip around the ring
      OP_REQUIRES_OK(context, context->GetAttr("ring_size", &ring_size_));
      OP_REQUIRES_OK(context, context->GetAttr("cluster_length", &cluster_length_));

      // should seed allows this node to seed a cluster if it has none
      // only one op in the ring should_seed to more closely replicate single thread results
      OP_REQUIRES_OK(context, context->GetAttr("should_seed", &should_seed_));

      // we keep processing until we have seen all the chunks
      OP_REQUIRES_OK(context, context->GetAttr("total_chunks", &total_chunks_));
      OP_REQUIRES_OK(context, context->GetAttr("chunk_size", &chunk_size_));
      
      OP_REQUIRES_OK(context, context->GetAttr("min_score", &params_.min_score));
      OP_REQUIRES_OK(context, context->GetAttr("subsequence_homology", &params_.subsequence_homology));
      OP_REQUIRES_OK(context, context->GetAttr("max_reps", &params_.max_representatives));
      OP_REQUIRES_OK(context, context->GetAttr("max_n_aa_not_covered", &params_.max_n_aa_not_covered));
     
      // for debugging
      OP_REQUIRES_OK(context, context->GetAttr("node_id", &node_id_));
      OP_REQUIRES_OK(context, context->GetAttr("do_allall", &do_allall_));

      num_chunks_ = 0;
    }

    Status Init(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &input_queue_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 1), &neighbor_queue_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 2), &neighbor_queue_out_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 3), &cluster_queue_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 4), &candidate_map_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 5), &executor_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "alignment_envs", &alignment_envs_container_));
      envs_ = alignment_envs_container_->get();
      //LOG(INFO) << "done init";
      return Status::OK();
    }

    ~AGDProteinClusterOp() {
      LOG(INFO) << abs_seq_.size() << " abs seqs from nb q";
      stringstream ss;
      for (auto i : abs_seq_) {
        ss << i << " ";
      }
      LOG(INFO) << ss.str();
      int total = 0;
      for (auto& cluster : clusters_) {
        total += cluster.TotalComps();
      }
      LOG(INFO) << "Node: " << node_id_ << " did a total of " << total << " 16b comps ";

      int len = 0;

      ofstream file;
      file.open(string("dump/clusters") + to_string(node_id_) + string(".json"));
      file << "[\n";
      int index = 1;
      for (auto& cluster : clusters_) {
        cluster.Dump(file);
        if (index == clusters_.size()) {
          file << "\n";
        } else {
          file << ",\n";
        }
        index++;
        if (cluster.LongestSeqLength() > len)
          len = cluster.LongestSeqLength();
      }
      file << "]\n";
      file.close();
  
      double seconds = double(total_wait_) / 1000000.0f;
      LOG(INFO) << "Node: " << node_id_ << " spent " << seconds << " waiting for input and had " 
        << clusters_.size() << " clusters, with a longest seq len of " << len;

    }


    // for debugz
    string PrintNormalizedProtein(const char* seq, size_t len) {
      scratch_.resize(len);
      memcpy(&scratch_[0], seq, len);
      for (size_t i = 0; i < len; i++) {
        scratch_[i] = scratch_[i] + 'A';
      }
      return string(&scratch_[0], len);
    }


    void Compute(OpKernelContext* ctx) override {
      // this entire thing is a bit spaghetti and could use a rewrite

      LOG(INFO) << "Node " << to_string(node_id_) << " Starting protein cluster";
      if (!neighbor_queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
        executor_->SetVars(candidate_map_, envs_, &params_);
      }

      if (num_chunks_ == total_chunks_*2) {
        // not 100% sure if this can happen yet
        LOG(INFO) << "Node " << to_string(node_id_) << " num chunk is TOTAL CHUNK!";
        ctx->SetStatus(errors::OutOfRange("cluster is done"));

        // the off chance that this node never seeded a cluster
        // and passed everything through. Possible on small datasets
        if (!cluster_queue_->is_closed()) {
          cluster_queue_->Close(ctx, false/*cancel pending enqueue*/, 
              [this](){ LOG(INFO) << "Node " << to_string(node_id_) <<"cluster queue closed"; } );
        }
        if (!neighbor_queue_out_->is_closed()) {
          neighbor_queue_out_->Close(ctx, false/*cancel pending enqueue*/, 
              [this](){ LOG(INFO) << "Node " << to_string(node_id_) <<"neighbor out queue closed"; } );
        }
        return;
      }

      /*pthread_attr_t attr;
      size_t stacksize;
      pthread_attr_init(&attr);
      pthread_attr_getstacksize(&attr, &stacksize);
      LOG(INFO) << "thread stack size is " << stacksize;*/
      // continues computing until it has seen all chunks
      // one chunk processed each Compute()
      // then outputs tensor encoded clusters

      Tensor chunk_t, num_recs_t, seq_number_t, was_added_t, sequence_t, coverages_t, genome_t, 
             first_ord_t, total_seqs_t, abs_seq_t;

      OP_REQUIRES_OK(ctx, DequeueChunk(ctx, chunk_t, num_recs_t, sequence_t, 
            was_added_t, coverages_t, genome_t, first_ord_t, total_seqs_t, abs_seq_t));
      //LOG(INFO) << "Node " << to_string(node_id_) << " dequeued a chunk ";

      auto sequence = sequence_t.scalar<int32>()();
      int32 new_sequence = sequence + 1;

      auto chunk = chunk_t.scalar<string>()();
      auto num_seqs = num_recs_t.scalar<int32>()();

      AGDRecordReader seqs_reader(chunk.data(), num_seqs);

      const char * data;
      size_t len;
      size_t i = 0;
      Status s = seqs_reader.GetNextRecord(&data, &len);


      Sketch data_sketch;
      Sketch::Parameters parameters;
      parameters.kmerSize = 3;              
      parameters.minHashesPerWindow = 1000; //sketch size
      parameters.noncanonical = true;
      setAlphabetFromString(parameters, alphabetProtein); //alphabetProtein declared in sketch.h
      char *Denormalised_data = Denormalise(data,len);
      data_sketch.init( Denormalised_data, len, "", "", parameters);

      OP_REQUIRES_OK(ctx, s);
      //LOG(INFO) << "Node " << to_string(node_id_) << " seq is " << sequence;
      
      auto was_added = was_added_t.vec<bool>();
      auto coverages = coverages_t.vec<string>();
      auto genome = genome_t.scalar<string>()();
      auto total_seqs = total_seqs_t.scalar<int32>()();
      auto first_ord = first_ord_t.scalar<int32>()();
      auto abs_seq = abs_seq_t.scalar<int32>()();
      auto genome_index = first_ord;
      
      //LOG(INFO) << "Node " << to_string(node_id_) << " dequeued abs seq " << abs_seq << " seq " << sequence << " ring size is " << ring_size_;
      //Sequence seq;
      //LOG(INFO) << "Node " << to_string(node_id_) << " genome : " << genome;
      int cluster_start = 0;

      if (sequence == ring_size_) {
        // this chunk has been evaluated by all nodes (made one round)
        // create new clusters for any non added proteins
        auto& point = abs_sequence_queue_.front();
        CHECK_EQ(point.first, abs_seq);
        cluster_start = point.second;
        abs_sequence_queue_.pop();
        
        LOG(INFO) << "Node " << to_string(node_id_) << " seen this chunk, dumping " << abs_seq 
          << " and creating cluster and Comparing to " << clusters_.size() - cluster_start << " more clusters";
       
        /*sequence_t.scalar<int32>()() = new_sequence;
        LOG(INFO) << "Node " << to_string(node_id_) << " neighbor queue size is " << neighbor_queue_out_->size();
        OP_REQUIRES_OK(ctx, EnqueueChunk(ctx, chunk_t, num_recs_t, sequence_t, was_added_t, coverages_t, genome_t, 
              first_ord_t, total_seqs_t, abs_seq_t));*/
        
        int num_compared = 0;
        MultiNotification n;
        int num_added = 0;
        while (s.ok()) {
          // first compare to all clusters prev created with a lower abs seq num
          for (size_t j = cluster_start; j < clusters_.size(); j++) {
            auto& cluster = clusters_[j];
            if (cluster.AbsoluteSequence() > abs_seq) continue;
            // fill seq and pass to evaluate
            num_compared++;
            Sequence s;
            s.data = data;
            s.length = len;
            s.coverages = &coverages(i);
            s.genome_index = genome_index;
            s.genome = &genome;
            s.total_seqs = total_seqs;
            s.data_sketch = data_sketch;

            auto item = make_tuple(s, &cluster, &was_added(i), &n);
            num_added++;
            /*auto added = cluster.EvaluateSequence(seq, envs_, &params_);

            if (!was_added(i)) was_added(i) = added;*/

            OP_REQUIRES_OK(ctx, executor_->EnqueueClusterEval(item));
          }
          s = seqs_reader.GetNextRecord(&data, &len);
          char *Denormalised_data = Denormalise(data,len);
          data_sketch.init( Denormalised_data, len, "", "", parameters);

          i++;
          genome_index++;
        }
        n.SetMinNotifies(num_added);
        n.WaitForNotification();
        //LOG(INFO) << "compared seqs to " << num_compared << " comparisons of lower or equal abs_seq ";

        genome_index = first_ord;
        i = 0;
        seqs_reader.Reset();
        s = seqs_reader.GetNextRecord(&data, &len);
        char *Denormalised_data = Denormalise(data,len);
        data_sketch.init( Denormalised_data, len, "", "", parameters);


        // now compare seqs within the chunk
        size_t start_cluster = clusters_.size();
        // create clusters
        while (s.ok()) {
          /*LOG(INFO) << "Node " << to_string(node_id_) << "was added " 
              << was_added(i) << " numuncovered: " << NumUncoveredAA(coverages(i));*/
        
          MultiNotification note;
          num_added = 0;
          if (start_cluster < clusters_.size()) {
            for (size_t j = start_cluster; j < clusters_.size(); j++) {
              auto& cluster = clusters_[j];
              if (cluster.AbsoluteSequence() > abs_seq) continue;
              // fill seq and pass to evaluate
              Sequence s;
              s.data = data;
              s.length = len;
              s.coverages = &coverages(i);
              s.genome_index = genome_index;
              s.genome = &genome;
              s.total_seqs = total_seqs;
              s.data_sketch = data_sketch;

              auto item = make_tuple(s, &cluster, &was_added(i), &note);
              num_added++;

              //auto added = cluster.EvaluateSequence(seq, envs_, &params_);

              //if (!was_added(i)) was_added(i) = added;
              OP_REQUIRES_OK(ctx, executor_->EnqueueClusterEval(item));
            }
            note.SetMinNotifies(num_added);
            note.WaitForNotification();
            // we have to wait here because we need to check if this seq
            // will form another cluster (below)
          }

          if (!was_added(i)) {
            // add cluster
            Cluster cluster(envs_, data, len, genome, genome_index, total_seqs, abs_seq, data_sketch);
            clusters_.push_back(std::move(cluster));
          } else if (params_.subsequence_homology && (NumUncoveredAA(coverages(i)) >
              params_.max_n_aa_not_covered)) {
            
            Cluster cluster(envs_, data, len, genome, genome_index, total_seqs, abs_seq, data_sketch);
            clusters_.push_back(std::move(cluster));

          }
          s = seqs_reader.GetNextRecord(&data, &len);
          char *Denormalised_data = Denormalise(data,len);
          data_sketch.init( Denormalised_data, len, "", "", parameters);
          i++;
          genome_index++;
        }

        sequence_t.scalar<int32>()() = new_sequence;
        LOG(INFO) << "Node " << to_string(node_id_) << " neighbor queue size is " << neighbor_queue_out_->size();
        OP_REQUIRES_OK(ctx, EnqueueChunk(ctx, chunk_t, num_recs_t, sequence_t, was_added_t, coverages_t, genome_t, 
              first_ord_t, total_seqs_t, abs_seq_t));
       
        //num_chunks_++;
        Tensor* out;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
        out->scalar<string>()() = "Node " + to_string(node_id_) + " execution is done";
        return;

      } else if (sequence == ring_size_ * 2) {  // chunk has made 2 rounds, done

        //LOG(INFO) << "chunk complete after two rounds -----------------";
        //LOG(INFO) << "Node " << to_string(node_id_) << " we have seen all " << num_chunks_ << " chunks, total is " << total_chunks_;
        LOG(INFO) << "Node " << to_string(node_id_) << "abs seq " << abs_seq << " has made two rounds ";

        if (num_chunks_ == total_chunks_ * 2) {
          // we have seen all chunks and are done, this chunk also started at this node
          // encode the clusters into tensors and enqueue them 
          // for downstream aggregation
          LOG(INFO) << "Node " << to_string(node_id_) << " we have seen all chunks, outputting clusters";
          LOG(INFO) << "Node " << to_string(node_id_) << "Total clusters: " << clusters_.size();

          //printing the number of sequences in each cluster
          for (int c = 0 ; c< clusters_.size(); c++){
           cout << "Cluster " << c <<  "Size: " << clusters_[c].NumSequences() << endl;
         }
          //
          // seqs in this chunk however have not been compared to one another

          if (do_allall_) {
          
            // notification
            // each cluster submit alignments, add num to notification min
            // wait
            // build output for each cluster
            MultiNotification n;
            int total = 0;
            for (auto& cluster : clusters_) {
              total += cluster.SubmitAlignments(executor_, &n);
            }
            //LOG(INFO) << "Node " << node_id_ << " submitted " << total << " alignments to queue";
            n.SetMinNotifies(total);
            n.WaitForNotification();

            for (auto& cluster : clusters_) {

              cluster.DoAllToAll(envs_, &params_);

              if (cluster.NumCandidates() == 0) continue; // no cands, dont bother

              vector<Tensor> ints, doubles, genomes;
              OP_REQUIRES_OK(ctx, cluster.BuildOutput(ints, doubles, genomes, cluster_length_, ctx)); // TODO adjust or parametrize '10'

              // enqueue tensor tuples into cluster queue
              for (size_t i = 0; i < ints.size(); i++) {
                OP_REQUIRES_OK(ctx, EnqueueClusters(ctx, ints[i], doubles[i], genomes[i]));
              }
            }
          }

          Tensor t;
          OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataType::DT_STRING, TensorShape({}), &t));
          t.scalar<string>()() = "Node " + to_string(node_id_) + " is done.";

          // close queues
          cluster_queue_->Close(ctx, false/*cancel pending enqueue*/, 
              [this](){ LOG(INFO) << "Node " << to_string(node_id_) <<"cluster queue closed"; } );
          neighbor_queue_out_->Close(ctx, false/*cancel pending enqueue*/, 
              [this](){ LOG(INFO) << "Node " << to_string(node_id_) <<"neighbor out queue closed"; } );
        }

        Tensor* out;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
        out->scalar<string>()() = "Node " + to_string(node_id_) + " execution is done";
        return;

      } else /*if (sequence != 0)*/ {
        //LOG(INFO) << " sequence not zerooooooooooooooooooooo! is " << sequence;
        // if < size, push to queue
        // if > size, must be first element in queue
        // remove from queue, and start from that cluster index 
        // queue<pair<int, int>> abs_sequence_queue_;
        if (sequence < ring_size_) { // first time we've seen
          abs_sequence_queue_.push(make_pair(abs_seq, clusters_.size()));
          LOG(INFO) << "pushing to queue";
          cluster_start = 0;
        } else if (sequence > ring_size_) { // second time we've seen
          auto& point = abs_sequence_queue_.front();
          CHECK_EQ(point.first, abs_seq);
          cluster_start = point.second;
          LOG(INFO) << "Node " << node_id_ << "received abs seq " << abs_seq << " for second time. " 
            << "Comparing to " << clusters_.size() - cluster_start << " more clusters";
          abs_sequence_queue_.pop();
        } else {
          LOG(INFO) << " equal to ring size wtf ";
          CHECK_EQ(0, 1);
        }
      }

      if (clusters_.empty() && should_seed_ && sequence == 0) {
        // seed a new cluster with first sequence
        LOG(INFO) << "Node " << to_string(node_id_) << " seeding cluster with sequence " 
          << PrintNormalizedProtein(data, len) << " genome: " << genome << " genome_index: " << genome_index 
          << " total_seqs: " << total_seqs;
       
        was_added(i) = true;

        // don't add coverages, to prevent the seed sequence 
        /*if (coverages(i).size() == 0)
          coverages(i).resize(len, 1);*/

        Cluster cluster(envs_, data, len, genome, genome_index, total_seqs, abs_seq, data_sketch);
        clusters_.push_back(std::move(cluster));
        // next sequence, and carry on
        s = seqs_reader.GetNextRecord(&data, &len);
        char *Denormalised_data = Denormalise(data,len);
        data_sketch.init( Denormalised_data, len, "", "", parameters);
        genome_index++;
        i++;

      } else if (clusters_.empty()) { // TODO not sure if this still needs to exist
        // pass this chunk to neighbor, 
        //LOG(INFO) << "Node " << to_string(node_id_) << " passing chunk to neighbor because no seed";
        
        sequence_t.scalar<int32>()() = new_sequence;
        LOG(INFO) << "Node " << to_string(node_id_) << " neighbor queue size is " << neighbor_queue_out_->size();
        OP_REQUIRES_OK(ctx, EnqueueChunk(ctx, chunk_t, num_recs_t, sequence_t, was_added_t, coverages_t, genome_t, 
              first_ord_t, total_seqs_t, abs_seq_t));
       
        //num_chunks_++;
        Tensor* out;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
        out->scalar<string>()() = "Node " + to_string(node_id_) + " execution is done";
        return;
      }
   
      bool passed = false;
      // this is some test logic for forwarding chunks so other nodes can process while 
      // we process
      /*if (sequence != ring_size_ - 1 && sequence != ring_size_*2 - 1) {

        // we pass early if the next node is not the originator of this chunk
        passed = true;
        // pass all to neighbor queue
        sequence_t.scalar<int32>()() = new_sequence;
        //LOG(INFO) << "Node " << to_string(node_id_) << " neighbor queue size is " << neighbor_queue_out_->size();
        //LOG(INFO) << "Node " << to_string(node_id_) << " enqueuing to neighbor seq " << new_sequence << " with abs seq " << abs_seq;
        OP_REQUIRES_OK(ctx, EnqueueChunk(ctx, chunk_t, num_recs_t, sequence_t, was_added_t, coverages_t, genome_t, 
              first_ord_t, total_seqs_t, abs_seq_t));
      }*/


      int num_added = 0;
      MultiNotification n;
      while (s.ok()) {
        /*LOG(INFO) << "Node " << to_string(node_id_) << " evaluating sequence " 
          << PrintNormalizedProtein(data, len)
          << " genome: " << genome << " genome_index: " << genome_index 
          << " total_seqs: " << total_seqs << ", seq len: " << len;*/


        if (coverages(i).size() == 0)
          coverages(i).resize(len, 1);

        for (size_t x = cluster_start; x < clusters_.size(); x++) {
          auto& cluster = clusters_[x];
          // fill seq and pass to evaluate
              
          if (cluster.AbsoluteSequence() > abs_seq) continue;

          Sequence s;
          s.data = data;
          s.length = len;
          s.coverages = &coverages(i);
          s.genome_index = genome_index;
          s.genome = &genome;
          s.total_seqs = total_seqs;
          s.data_sketch = data_sketch;

          // workitem with seq*, cluster*, bool*, multinotification*
          auto item = make_tuple(s, &cluster, &was_added(i), &n);
          num_added++;
          
          OP_REQUIRES_OK(ctx, executor_->EnqueueClusterEval(item));
          
        }

        s = seqs_reader.GetNextRecord(&data, &len);
        char *Denormalised_data = Denormalise(data,len);
        data_sketch.init( Denormalised_data, len, "", "", parameters);
        genome_index++;
        i++;
      }
      n.SetMinNotifies(num_added);
      n.WaitForNotification();
      LOG(INFO) << "finished " << num_added << " cluster evals";

      if (!passed) {
        // pass all to neighbor queue
        sequence_t.scalar<int32>()() = new_sequence;
        //LOG(INFO) << "Node " << to_string(node_id_) << " neighbor queue size is " << neighbor_queue_out_->size();
        //LOG(INFO) << "Node " << to_string(node_id_) << " enqueuing to neighbor seq " << new_sequence << " with abs seq " << abs_seq;
        OP_REQUIRES_OK(ctx, EnqueueChunk(ctx, chunk_t, num_recs_t, sequence_t, was_added_t, coverages_t, genome_t, 
              first_ord_t, total_seqs_t, abs_seq_t));
      }

      //LOG(INFO) << "Node " << to_string(node_id_) << " Total clusters: " << clusters_.size();

      if (num_chunks_ == total_chunks_ * 2) {
        // we have seen all chunks and are done, last chunk did not start at this node
        // encode the clusters into tensors and enqueue them 
        // for downstream aggregation
        LOG(INFO) << "Node " << to_string(node_id_) << " we have seen all chunks, outputting clusters";
        LOG(INFO) << "Node " << to_string(node_id_) << "Total clusters: " << clusters_.size();

        if (do_allall_) {
          MultiNotification n;
          int total = 0;
          for (auto& cluster : clusters_) {
            total += cluster.SubmitAlignments(executor_, &n);
          }
          //LOG(INFO) << "Node " << node_id_ << " submitted " << total << " alignments to queue";
          n.SetMinNotifies(total);
          n.WaitForNotification();

          for (auto& cluster : clusters_) {

            cluster.DoAllToAll(envs_, &params_);

            if (cluster.NumCandidates() == 0) continue; // no cands, dont bother

            vector<Tensor> ints, doubles, genomes;
            OP_REQUIRES_OK(ctx, cluster.BuildOutput(ints, doubles, genomes, cluster_length_, ctx)); // TODO adjust or parametrize '10'

            // enqueue tensor tuples into cluster queue
            for (size_t i = 0; i < ints.size(); i++) {
              OP_REQUIRES_OK(ctx, EnqueueClusters(ctx, ints[i], doubles[i], genomes[i]));
            }
          }
        }

        Tensor t;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataType::DT_STRING, TensorShape({}), &t));
        t.scalar<string>()() = "Node " + to_string(node_id_) + " is done.";

        // close queues
        cluster_queue_->Close(ctx, false/*cancel pending enqueue*/, 
            [this](){ LOG(INFO) << "Node " << to_string(node_id_) <<"cluster queue closed"; } );
        neighbor_queue_out_->Close(ctx, false/*cancel pending enqueue*/, 
            [this](){ LOG(INFO) << "Node " << to_string(node_id_) <<"neighbor out queue closed"; } );
      }

      //num_chunks_++;
      Tensor* out;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
      out->scalar<string>()() = "Node " + to_string(node_id_) + " execution is done";

      //LOG(INFO) << "DONE running mark duplicates!! Found so far: " << num_dups_found_;

    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;
    QueueInterface *neighbor_queue_ = nullptr;
    QueueInterface *neighbor_queue_out_ = nullptr;
    QueueInterface *input_queue_ = nullptr;
    QueueInterface *cluster_queue_ = nullptr;
    bool input_closed = false;
    BasicContainer<AlignmentEnvironments> *alignment_envs_container_ = nullptr;
    Parameters params_;
    AlignmentEnvironments* envs_;
    int chunk_size_ = 0;
    int cluster_length_ = 0;

    vector<Cluster> clusters_;

    // record that we have alignments (candidate) between specific (g1, g2) (s1, s2) pairs
    // will still have duplicates between ring nodes
    // TODO make this into a locked shared resource
    CandidateMap* candidate_map_ = nullptr;

    // queue of pairs of (abs_seq, cluster_size) used to prevent comparing
    // second round chunks to clusters < cluster_size
    // i.e. we only want to compare to clusters that have been added since 
    // we last saw a chunk
    queue<pair<int, int>> abs_sequence_queue_;

    int ring_size_;
    bool should_seed_ = false;
    int total_chunks_;
    int num_chunks_;
    vector<char> scratch_;
    int node_id_;
    vector<int> abs_seq_;
    int64 total_wait_ = 0;
    bool do_allall_ = true;

    AlignmentExecutor* executor_;

    int NumUncoveredAA(const string& coverages) {
      int sum = 0;
      for (size_t i = 0; i < coverages.size(); i++)
        sum += (int)coverages[i];
      return sum;
    }

    inline Status DequeueChunk(OpKernelContext* ctx, Tensor& chunk, Tensor& num_recs, 
        Tensor& sequence, Tensor& was_added, Tensor& coverages, Tensor& genome, Tensor& first_ord,
        Tensor& total_seqs, Tensor& abs_seq) {
        
      LOG(INFO) << "Node " << to_string(node_id_) << " input queue closed " << input_queue_->is_closed() 
        << "input queue size: " << input_queue_->size();

      auto begin = std::chrono::steady_clock::now();
      // prefer to dequeue neighbor queue, otherwise attempt to dequeue the main input
      if (neighbor_queue_->size() > 0) {
      
        //LOG(INFO) << "Node " << to_string(node_id_) << " neighbor size is " << neighbor_queue_->size();
        // dequeue neighbor
      } else if (input_queue_->size() > 0 || !input_queue_->is_closed()) {
        // dequeue input
        //LOG(INFO) << "Node " << to_string(node_id_) << " dequeueing input";
        
        Notification n;
        input_queue_->TryDequeue(ctx, [this, ctx, &n, &chunk, &num_recs, &sequence, 
            &was_added, &coverages, &genome, &first_ord, &total_seqs, &abs_seq](const QueueInterface::Tuple &tuple) {
            if (!ctx->status().ok()) {
              LOG(INFO) << "Node " << node_id_ << " input queue dequeue FAILED " << ctx->status();
              n.Notify(); // should only happen when input queue closes
              return;
            }
            chunk = tuple[0];
            num_recs = tuple[1];
            genome = tuple[5];
            first_ord = tuple[6];
            total_seqs = tuple[7];
            abs_seq = tuple[8];
            
            /*sequence = tuple[2];
            was_added = tuple[3];
            coverages = tuple[4];*/
            n.Notify();
        });
        LOG(INFO) << "Node " << node_id_ << " waiting on input queue.";
        n.WaitForNotification();
        LOG(INFO) << "Node " << node_id_ << " done.";
       
        // it's possible another node took the last, and we received
        // an OutOfRange or Cancelled status.
        // Queue is closed, return and wait for reexecution
        if (!ctx->status().ok()) {
          LOG(INFO) << "INPUT QUEUE FAILED WITH " << ctx->status();
          return ctx->status();
        }

        // initialize extra data for this chunk
        TF_RETURN_IF_ERROR(ctx->allocate_temp(DataType::DT_INT32, TensorShape({}), &sequence));
        sequence.scalar<int32>()() = 0;
        TF_RETURN_IF_ERROR(ctx->allocate_temp(DataType::DT_BOOL, TensorShape({chunk_size_}), &was_added));
        auto wa = was_added.vec<bool>();
        for (size_t i = 0; i < chunk_size_; i++)
          wa(i) = false;
        TF_RETURN_IF_ERROR(ctx->allocate_temp(DataType::DT_STRING, TensorShape({chunk_size_}), &coverages));
        // else we cont below and dequeue neighbor
        //LOG(INFO) << "Node " << node_id_ << " status was " << ctx->status() << " continuing to neighbor ";
      
        auto end = std::chrono::steady_clock::now();
        total_wait_ += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        return Status::OK();
      } 

      // dequeue neighbor, and wait
      //LOG(INFO) << "Node " << to_string(node_id_) << " dequeueing neighbor";
      Notification n;
      neighbor_queue_->TryDequeue(ctx, [this, ctx, &n, &chunk, &num_recs, &sequence, 
          &was_added, &coverages, &genome, &first_ord, &total_seqs, &abs_seq](const QueueInterface::Tuple &tuple) {
          if (!ctx->status().ok()) {
            LOG(INFO) << "Node " << node_id_ << " neighbor queue closed, getting last chunk?"; //should never happen
            //n.Notify(); 
            //return;
          }
          chunk = tuple[0];
          num_recs = tuple[1];
          sequence = tuple[2];
          was_added = tuple[3];
          coverages = tuple[4];
          genome = tuple[5];
          first_ord = tuple[6];
          total_seqs = tuple[7];
          abs_seq = tuple[8];
          n.Notify();
      });
      LOG(INFO) << "Node " << node_id_ << " waiting on neighbor queue.";
      n.WaitForNotification();
      LOG(INFO) << "Node " << to_string(node_id_) << " done.";
      num_chunks_++;
      auto abs_seq_val = abs_seq.scalar<int32>()();
      abs_seq_.push_back(abs_seq_val);
      auto end = std::chrono::steady_clock::now();
      total_wait_ += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
      return Status::OK();
    }

    inline Status EnqueueChunk(OpKernelContext *ctx, const Tensor& chunk, 
        const Tensor& num_recs, const Tensor& sequence, const Tensor& was_added, 
        const Tensor& coverages, const Tensor& genome, const Tensor& first_ord, 
        const Tensor& total_seqs, const Tensor& abs_seq) {
      Notification n;
      QueueInterface::Tuple tuple;
      tuple.resize(9);
      tuple[0] = chunk; tuple[1] = num_recs;
      tuple[2] = sequence; tuple[3] = was_added;
      tuple[4] = coverages; tuple[5] = genome;
      tuple[6] = first_ord; tuple[7] = total_seqs;
      tuple[8] = abs_seq;
      neighbor_queue_out_->TryEnqueue(tuple, ctx, [this, &n, ctx]() {
          if (!ctx->status().ok()) {
            LOG(INFO) << "Node " << node_id_ << " neighbor queue enqueue FAILED " << ctx->status(); //should never happen
            //n.Notify(); 
            //return;
          }
          n.Notify();
      });
      n.WaitForNotification();
      return Status::OK();
    }
    
    inline Status EnqueueClusters(OpKernelContext *ctx, const Tensor& ints,
        const Tensor& doubles, const Tensor& genomes) {
      Notification n;
      QueueInterface::Tuple tuple;
      tuple.resize(3);
      tuple[0] = ints;
      tuple[1] = doubles;
      tuple[2] = genomes;
      cluster_queue_->TryEnqueue(tuple, ctx, [&n]() {
          n.Notify();
        });
      n.WaitForNotification();
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDProteinCluster").Device(DEVICE_CPU), AGDProteinClusterOp);
} //  namespace tensorflow {
