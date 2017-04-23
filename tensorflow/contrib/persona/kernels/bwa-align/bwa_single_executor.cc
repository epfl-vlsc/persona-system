//
// Created by Stuart Byma on 17/04/17.
//

#include "tensorflow/contrib/persona/kernels/bwa-align/bwa_single_executor.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  BWASingleExecutor::BWASingleExecutor(Env *env, bwaidx_t *index, mem_opt_t *options,
                                       int max_secondary, int num_threads, int capacity,
                                       int max_read_size) : index_(index),
                                                            options_(options),
                                                            num_threads_(num_threads),
                                                            capacity_(capacity),
                                                            max_secondary_(max_secondary),
                                                            max_read_size_(max_read_size) {
    // create a threadpool to execute stuff
    workers_.reset(new thread::ThreadPool(env, "BWASingle", num_threads_));
    request_queue_.reset(new ConcurrentQueue<std::shared_ptr<ResourceContainer<BWAReadResource>>>(capacity));
    init_workers();
  }

  BWASingleExecutor::~BWASingleExecutor() {
    if (!run_) {
      LOG(ERROR) << "Unable to safely wait in ~BWAAlignSingleOp for all threads. run_ was toggled to false\n";
    }
    run_ = false;
    request_queue_->unblock();
    while (num_active_threads_.load(std::memory_order_relaxed) > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  Status BWASingleExecutor::EnqueueChunk(std::shared_ptr<ResourceContainer<BWAReadResource> > chunk) {
    if (!compute_status_.ok()) return compute_status_;
    if (!request_queue_->push(chunk))
      return Internal("Single executor failed to push to request queue");
    else
      return Status::OK();
  }

  void BWASingleExecutor::init_workers() {

    auto aligner_func = [this]() {
      //std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
      int my_id = 0;
      {
        mutex_lock l(mu_);
        my_id = id_.fetch_add(1, memory_order_relaxed);
      }

      bwa_wrapper::BWAAligner aligner(options_, index_, max_read_size_);
      vector<AlignmentResultBuilder> result_builders;
      result_builders.resize(max_secondary_+1);

      vector<BufferPair*> result_bufs;
      result_bufs.reserve(max_secondary_+1);
      ReadResource* subchunk_resource = nullptr;
      Status io_chunk_status, subchunk_status;
      //std::chrono::high_resolution_clock::time_point end_subchunk = std::chrono::high_resolution_clock::now();
      //std::chrono::high_resolution_clock::time_point start_subchunk = std::chrono::high_resolution_clock::now();

      while (run_) {
        // reads must be in this scope for the custom releaser to work!
        shared_ptr<ResourceContainer<BWAReadResource>> reads_container;
        if (!request_queue_->peek(reads_container)) {
          continue;
        }
        //timeLog.peek = std::chrono::high_resolution_clock::now();

        auto *reads = reads_container->get();

        size_t interval;
        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs, &interval);
        while (io_chunk_status.ok()) {
          //LOG(INFO) << "finalizer thread " << my_id << " got  interval: " << interval;

          for (int i = 0; i < result_builders.size(); i++)
            result_builders[i].SetBufferPair(result_bufs[i]);

          Status s = aligner.AlignSubchunkSingle(subchunk_resource, result_builders);

          if (!s.ok()){
            compute_status_ = s;
            return;
          }

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs, &interval);
        }

        if (!IsResourceExhausted(io_chunk_status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError for I/O Chunk! : " << io_chunk_status << "\n";
          compute_status_ = io_chunk_status;
          return;
        }

        request_queue_->drop_if_equal(reads_container);

      }

      //VLOG(INFO) << "base aligner thread ending.";
      num_active_threads_--;

    };
    for (int i = 0; i < num_threads_; i++)
      workers_->Schedule(aligner_func);
    num_active_threads_ = num_threads_;
  }

}
