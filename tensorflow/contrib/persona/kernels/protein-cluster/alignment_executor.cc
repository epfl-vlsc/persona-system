

#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_executor.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/protein_cluster.h"
#include <pthread.h>

namespace tensorflow {

using namespace std;
using namespace errors;

void OutputThreadStackSize() {
    
    pthread_attr_t attr;
    int s = pthread_getattr_np(pthread_self(), &attr);
    size_t stack_size;
    s = pthread_attr_getstacksize(&attr, &stack_size);
    LOG(INFO) << "stack size is " << stack_size << " bytes ";

}

AlignmentExecutor::AlignmentExecutor(Env *env, int num_threads, int num_threads_align, int capacity) : 
  num_threads_(num_threads),
  num_threads_align_(num_threads_align),
  capacity_(capacity) {

  // num_threads is for cluster evaluators
  // num_threads_align is for final aligners
  // create a threadpool to execute stuff
  workers_.reset(new thread::ThreadPool(env, "AlignmentExecutor", num_threads_align_));
  eval_workers_.reset(new thread::ThreadPool(env, "ClusterAlignmentExecutor", num_threads_));
  work_queue_.reset(new ConcurrentQueue<WorkItem>(capacity));
  cluster_work_queue_.reset(new ConcurrentQueue<ClusterWorkItem>(capacity));
  init_workers();

}

AlignmentExecutor::~AlignmentExecutor() {
  if (!run_) {
    LOG(ERROR) << "Unable to safely wait in ~AlignmentExecutor for all threads. run_ was toggled to false\n";
  }
  run_ = false;
  work_queue_->unblock();
  while (num_active_threads_.load(std::memory_order_relaxed) > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void AlignmentExecutor::init_workers() {

  auto aligner_func = [this]() {
    //std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    int my_id = id_.fetch_add(1, memory_order_relaxed);
    LOG(INFO) << "Alignment thread spinning up with id " << my_id;

    int capacity = work_queue_->capacity();

    ProteinAligner aligner(envs_, params_);

    while (run_) {
      // read from queue, and align work item 
      WorkItem item;
      if (!work_queue_->pop(item)) {
        continue;
      }


      if (!aligner.Params()) {
        aligner = ProteinAligner(envs_, params_);
      }

      //LOG(INFO) << my_id << " got alignment from queue ";
      const ClusterSequence* seq1 = nullptr;
      const ClusterSequence* seq2 = nullptr;
      seq1 = get<0>(item); seq2 = get<1>(item);

      if (seq1->TotalSeqs() > seq2->TotalSeqs() || 
          ((seq1->TotalSeqs() == seq2->TotalSeqs()) && seq1->Genome() > 
           seq2->Genome())) {
        //seq1 = sequence;
        std::swap(seq1, seq2);
      } else {
        //seq2 = sequence;
      }
      
      ProteinAligner::Alignment& alignment = *get<2>(item);
      alignment.score = 0; // 0 score will signify not to create candidate

      if (seq1->Genome() == seq2->Genome() && seq1->GenomeIndex() == seq2->GenomeIndex()) {
        MultiNotification* n = get<3>(item);
        n->Notify();
        continue;
      }

      if (seq1->Genome() == seq2->Genome() && seq1->GenomeIndex() > seq2->GenomeIndex()) {
        std::swap(seq1, seq2);
      }

      auto genome_pair = make_pair(seq1->Genome(), seq2->Genome());
      auto seq_pair = make_pair(seq1->GenomeIndex(), seq2->GenomeIndex());
      if (candidate_map_ && !candidate_map_->ExistsOrInsert(genome_pair, seq_pair)) {

        if (aligner.PassesThreshold(seq1->Data(), seq2->Data(), 
              seq1->Length(), seq2->Length())) {

          Status s = aligner.AlignLocal(seq1->Data(), seq2->Data(), 
              seq1->Length(), seq2->Length(), alignment);
        }
      }

      MultiNotification* n = get<3>(item);
      n->Notify();
      total_alignments_++;
      /*if (total_alignments_.load() % 1000 == 0)
        LOG(INFO) << "total alignments " << total_alignments_.load();*/

      auto compute_error = !compute_status_.ok();
      if (compute_error) {
        LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError! : " << compute_status_
                   << "\n";
        run_ = false;
        break;
      }
    }

    VLOG(INFO) << "aligner executor thread ending.";
    num_active_threads_.fetch_sub(1, memory_order_relaxed);
  };

  num_active_threads_ = num_threads_align_;
  for (int i = 0; i < num_threads_align_; i++)
    workers_->Schedule(aligner_func);
  
  auto cluster_func = [this]() {
    //std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    int my_id = id_.fetch_add(1, memory_order_relaxed);
    LOG(INFO) << "Cluster eval thread spinning up with id " << my_id;

    ClusterWorkItem item;
    while (run_) {
      // read from queue, and align work item 
      if (!cluster_work_queue_->pop(item)) {
        continue;
      }
      
      /*if (work_queue_->size() == 0) {
        num_times_emptied_++;
        if (num_times_emptied_.load() % 1000 == 0)
          LOG(INFO) << "total times emptied " << num_times_emptied_.load();
      }*/

      auto& seq = get<0>(item);
      auto cluster = get<1>(item);
      auto& was_added = *get<2>(item);
      auto n = get<3>(item);
            
      auto added = cluster->EvaluateSequence(seq, envs_, params_);

      if (!was_added) was_added = added;

      n->Notify(); 
      
      auto compute_error = !compute_status_.ok();
      if (compute_error) {
        LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError! : " << compute_status_
                   << "\n";
        run_ = false;
        break;
      }
    }
    
    VLOG(INFO) << "cluster eval executor thread ending.";
    num_active_threads_.fetch_sub(1, memory_order_relaxed);
  };
  
  num_active_threads_ += num_threads_;
  for (int i = 0; i < num_threads_; i++)
    eval_workers_->Schedule(cluster_func);

  LOG(INFO) << "Finished spinning up worker threads.";
}

Status AlignmentExecutor::EnqueueAlignment(const WorkItem& item) {

  if (!work_queue_->push(item)) {
    return errors::Internal("Failed to push to alignment work queue");
  } else { 
    return Status::OK();
  }

}
    
Status AlignmentExecutor::EnqueueClusterEval(const ClusterWorkItem& item) {

  if (!cluster_work_queue_->push(item)) {
    return errors::Internal("Failed to push to alignment work queue");
  } else { 
    return Status::OK();
  }
}

}
