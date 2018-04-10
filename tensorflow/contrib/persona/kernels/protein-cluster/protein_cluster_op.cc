
#include <cstdint>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/params.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/cluster.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_environment.h"

namespace tensorflow {

   namespace { 
      void resource_releaser(ResourceContainer<Data> *data) {
        core::ScopedUnref a(data);
        data->release();
      }
   }


  using namespace std;
  using namespace errors;
  using namespace format;

  class AGDProteinClusterOp : public OpKernel {
  public:
    AGDProteinClusterOp(OpKernelConstruction *context) : OpKernel(context) {
      // ring ID is compared to sequence ID of incoming chunks to see if we have processed it before
      // indicating it has made a full trip around the ring
      OP_REQUIRES_OK(context, context->GetAttr("ring_size", &ring_size_));

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

      num_chunks_ = 0;
    }

    Status Init(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &input_queue_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 1), &neighbor_queue_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 2), &neighbor_queue_out_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 3), &cluster_queue_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "alignment_envs", &alignment_envs_container_));
      envs_ = alignment_envs_container_->get();
      LOG(INFO) << "done init";
      return Status::OK();
    }

    string PrintNormalizedProtein(const char* seq, size_t len) {
      scratch_.resize(len);
      memcpy(&scratch_[0], seq, len);
      for (size_t i = 0; i < len; i++) {
        scratch_[i] = scratch_[i] + 'A';
      }
      return string(&scratch_[0], len);
    }


    void Compute(OpKernelContext* ctx) override {
      LOG(INFO) << "Node " << to_string(node_id_) << " Starting protein cluster";
      if (!neighbor_queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      if (num_chunks_ == total_chunks_) {
        // not 100% sure if this can happen yet
        // just pass to neighbor because we are done
        LOG(INFO) << "Node " << to_string(node_id_) << " num chunk is TOTAL CHUNK!";
        ctx->SetStatus(errors::OutOfRange("cluster is done"));
        return;
      }

      // continues computing until it has seen all chunks
      // one chunk processed each Compute()
      // then outputs tensor encoded clusters

      Tensor chunk_t, num_recs_t, seq_number_t, was_added_t, sequence_t, coverages_t, genome_t, 
             first_ord_t, total_seqs_t;

      OP_REQUIRES_OK(ctx, DequeueChunk(ctx, chunk_t, num_recs_t, sequence_t, 
            was_added_t, coverages_t, genome_t, first_ord_t, total_seqs_t));
      LOG(INFO) << "Node " << to_string(node_id_) << " dequeued a chunk ";

      auto sequence = sequence_t.scalar<int32>()();
      int32 new_sequence = sequence + 1;

      auto chunk = chunk_t.scalar<string>()();
      auto num_seqs = num_recs_t.scalar<int32>()();

      AGDRecordReader seqs_reader(chunk.data(), num_seqs);

      const char * data;
      size_t len;
      size_t i = 0;
      Status s = seqs_reader.GetNextRecord(&data, &len);
      //LOG(INFO) << "Node " << to_string(node_id_) << " seq reader stat is  " << s;

      OP_REQUIRES_OK(ctx, s);
      LOG(INFO) << "Node " << to_string(node_id_) << " seq is " << sequence;
      
      auto was_added = was_added_t.vec<bool>();
      auto coverages = coverages_t.vec<string>();
      auto genome = genome_t.scalar<string>()();
      auto total_seqs = total_seqs_t.scalar<int32>()();
      auto first_ord = first_ord_t.scalar<int32>()();
      auto genome_index = first_ord;

      if (sequence == ring_size_) {
        // this chunk has been evaluated by all nodes
        // create new clusters for any non added proteins
        LOG(INFO) << "Node " << to_string(node_id_) << " seen this chunk, dumping " + to_string(sequence) + " and creating clusters";

        // create clusters
        while (s.ok()) {
          LOG(INFO) << "Node " << to_string(node_id_) << "was added " 
              << was_added(i) << " numuncovered: " << NumUncoveredAA(coverages(i));
          if (!was_added(i) || params_.subsequence_homology && NumUncoveredAA(coverages(i)) >
              params_.max_n_aa_not_covered) {
            // add cluster
            LOG(INFO) << "Node " << to_string(node_id_) << " creating cluster ";
            Cluster cluster(envs_, data, len, genome, genome_index, total_seqs);
            clusters_.push_back(std::move(cluster));
          } 
          s = seqs_reader.GetNextRecord(&data, &len);
          i++;
          genome_index++;
        }

        Tensor* out;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
        out->scalar<string>()() = "Node " + to_string(node_id_) + " execution is done";
        return;

      }

      if (clusters_.empty() && should_seed_) {
        // seed a new cluster with first sequence
        LOG(INFO) << "Node " << to_string(node_id_) << " seeding cluster with sequence " 
          << PrintNormalizedProtein(data, len) << " genome: " << genome << " genome_index: " << genome_index 
          << " total_seqs: " << total_seqs;
       
        was_added(i) = true;

        // don't add coverages, to prevent the seed sequence 
        /*if (coverages(i).size() == 0)
          coverages(i).resize(len, 1);*/

        Cluster cluster(envs_, data, len, genome, genome_index, total_seqs);
        clusters_.push_back(std::move(cluster));
        // next sequence, and carry on
        s = seqs_reader.GetNextRecord(&data, &len);
        genome_index++;
        i++;

      } else if (clusters_.empty()) {
        // pass this chunk to neighbor, 
        sequence_t.scalar<int32>()() = new_sequence;
        OP_REQUIRES_OK(ctx, EnqueueChunk(ctx, chunk_t, num_recs_t, sequence_t, was_added_t, coverages_t, genome_t, 
              first_ord_t, total_seqs_t));
        
        Tensor* out;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
        out->scalar<string>()() = "Node " + to_string(node_id_) + " execution is done";
        return;
      }
     
      Sequence seq;

      while (s.ok()) {
        LOG(INFO) << "Node " << to_string(node_id_) << " evaluating sequence " 
          << PrintNormalizedProtein(data, len)
          << " genome: " << genome << " genome_index: " << genome_index 
          << " total_seqs: " << total_seqs;


        if (coverages(i).size() == 0)
          coverages(i).resize(len, 1);

        for (auto& cluster : clusters_) {
          // fill seq and pass to evaluate
          seq.data = data;
          seq.length = len;
          seq.coverages = &coverages(i);
          seq.genome_index = genome_index;
          seq.genome = &genome;
          seq.total_seqs = total_seqs;

          auto added = cluster.EvaluateSequence(seq, envs_, &params_);

          if (!was_added(i)) was_added(i) = added;
          
          if (added) {
            LOG(INFO) << "Node " << to_string(node_id_) << " added sequence to cluster\n" 
              << "coverages is now " << coverages;
          }
        }

        s = seqs_reader.GetNextRecord(&data, &len);
        genome_index++;
        i++;
      }

      // pass all to neighbor queue
      sequence_t.scalar<int32>()() = new_sequence;
      OP_REQUIRES_OK(ctx, EnqueueChunk(ctx, chunk_t, num_recs_t, sequence_t, was_added_t, coverages_t, genome_t, 
            first_ord_t, total_seqs_t));

      if (++num_chunks_ == total_chunks_) {
        // we have seen all chunks and are done, 
        // encode the clusters into tensors and enqueue them 
        // for downstream aggregation
        LOG(INFO) << "Node " << to_string(node_id_) << " we have seen all chunks, outputting clusters";
        
        for (auto& cluster : clusters_) {
          vector<Tensor> ints, doubles, genomes;
          OP_REQUIRES_OK(ctx, cluster.BuildOutput(ints, doubles, genomes, 10, ctx));

          // enqueue tensor tuples into cluster queue
        }

        Tensor t;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataType::DT_STRING, TensorShape({}), &t));
        t.scalar<string>()() = "Node " + to_string(node_id_) + " is done.";
        OP_REQUIRES_OK(ctx, EnqueueClusters(ctx, t));

        // close queues
        cluster_queue_->Close(ctx, false/*cancel pending enqueue*/, 
            [this](){ LOG(INFO) << "Node " << to_string(node_id_) <<"cluster queue closed"; } );
        neighbor_queue_out_->Close(ctx, false/*cancel pending enqueue*/, 
            [this](){ LOG(INFO) << "Node " << to_string(node_id_) <<"neighbor out queue closed"; } );
      }

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

    vector<Cluster> clusters_;
  
    int ring_size_;
    bool should_seed_ = false;
    int total_chunks_;
    int num_chunks_;
    vector<char> scratch_;
    int node_id_;

    int NumUncoveredAA(const string& coverages) {
      int sum = 0;
      for (size_t i = 0; i < coverages.size(); i++)
        sum += (int)coverages[i];
      return sum;
    }
     
    inline Status DequeueChunk(OpKernelContext* ctx, Tensor& chunk, Tensor& num_recs, 
        Tensor& sequence, Tensor& was_added, Tensor& coverages, Tensor& genome, Tensor& first_ord,
        Tensor& total_seqs) {
        
      LOG(INFO) << "Node " << to_string(node_id_) << " dequeueing ...";
      LOG(INFO) << "Node " << to_string(node_id_) << " input queue is closed " << input_queue_->is_closed() 
        << "input queue size: " << input_queue_->size();

      // prefer to dequeue neighbor queue, otherwise attempt to dequeue the main input
      if (neighbor_queue_->size() > 0) {
      
        LOG(INFO) << "Node " << to_string(node_id_) << " neighbor size is " << neighbor_queue_->size();
        // dequeue neighbor
      } else if (!input_queue_->is_closed() || input_queue_->is_closed() && input_queue_->size() > 0) {
        // dequeue input
        LOG(INFO) << "Node " << to_string(node_id_) << " dequeueing input";
        
        Notification n;
        input_queue_->TryDequeue(ctx, [ctx, &n, &chunk, &num_recs, &sequence, 
            &was_added, &coverages, &genome, &first_ord, &total_seqs](const QueueInterface::Tuple &tuple) {
            if (!ctx->status().ok()) {
              n.Notify(); // should only happen when input queue closes
              return;
            }
            chunk = tuple[0];
            num_recs = tuple[1];
            genome = tuple[5];
            first_ord = tuple[6];
            total_seqs = tuple[7];
            
            /*sequence = tuple[2];
            was_added = tuple[3];
            coverages = tuple[4];*/
            n.Notify();
        });
        n.WaitForNotification();
        TF_RETURN_IF_ERROR(ctx->allocate_temp(DataType::DT_INT32, TensorShape({}), &sequence));
        sequence.scalar<int32>()() = 0;
        TF_RETURN_IF_ERROR(ctx->allocate_temp(DataType::DT_BOOL, TensorShape({chunk_size_}), &was_added));
        auto wa = was_added.vec<bool>();
        for (size_t i = 0; i < chunk_size_; i++)
          wa(i) = false;
        TF_RETURN_IF_ERROR(ctx->allocate_temp(DataType::DT_STRING, TensorShape({chunk_size_}), &coverages));
        if (ctx->status().ok()) {
          return Status::OK();
        } else if (!errors::IsOutOfRange(ctx->status()))
          return ctx->status();
        // else we cont below and dequeue neighbor
      } 

      // dequeue neighbor, and wait
      LOG(INFO) << "Node " << to_string(node_id_) << " dequeueing neighbor";
      Notification n;
      neighbor_queue_->TryDequeue(ctx, [ctx, &n, &chunk, &num_recs, &sequence, 
          &was_added, &coverages, &genome, &first_ord, &total_seqs](const QueueInterface::Tuple &tuple) {
          if (!ctx->status().ok()) {
            LOG(INFO) << "neighbor queue closed WTFFFFFF"; //should never happen
            n.Notify(); 
            return;
          }
          chunk = tuple[0];
          num_recs = tuple[1];
          sequence = tuple[2];
          was_added = tuple[3];
          coverages = tuple[4];
          genome = tuple[5];
          first_ord = tuple[6];
          total_seqs = tuple[7];
          n.Notify();
      });
      n.WaitForNotification();
      return Status::OK();
    }

    inline Status EnqueueChunk(OpKernelContext *ctx, const Tensor& chunk, 
        const Tensor& num_recs, const Tensor& sequence, const Tensor& was_added, 
        const Tensor& coverages, const Tensor& genome, const Tensor& first_ord, 
        const Tensor& total_seqs) {
      Notification n;
      QueueInterface::Tuple tuple;
      tuple.resize(8);
      tuple[0] = chunk; tuple[1] = num_recs;
      tuple[2] = sequence; tuple[3] = was_added;
      tuple[4] = coverages; tuple[5] = genome;
      tuple[6] = first_ord; tuple[7] = total_seqs;
      neighbor_queue_out_->TryEnqueue(tuple, ctx, [&n]() {
          n.Notify();
        });
      n.WaitForNotification();
      return Status::OK();
    }
    
    inline Status EnqueueClusters(OpKernelContext *ctx, const Tensor& stuff) {
      Notification n;
      QueueInterface::Tuple tuple;
      tuple.resize(1);
      tuple[0] = stuff;
      cluster_queue_->TryEnqueue(tuple, ctx, [&n]() {
          n.Notify();
        });
      n.WaitForNotification();
      return Status::OK();
    }


  };

  REGISTER_KERNEL_BUILDER(Name("AGDProteinCluster").Device(DEVICE_CPU), AGDProteinClusterOp);
} //  namespace tensorflow {
