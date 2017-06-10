//
// Created by Stuart Byma on 10/06/17.
//

#include "tensorflow/contrib/persona/kernels/bwa-align/bwa_paired_executor.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  BWAPairedExecutor::BWAPairedExecutor(Env *env, bwaidx_t *index, mem_opt_t *options,
                                       int max_secondary, int num_threads, int capacity,
                                       int max_read_size, float thread_ratio) : index_(index),
                                                            options_(options),
                                                            num_threads_(num_threads),
                                                            capacity_(capacity),
                                                            max_secondary_(max_secondary),
                                                            thread_ratio_(thread_ratio),
                                                            max_read_size_(max_read_size) {
    // create a threadpool to execute stuff
    workers_.reset(new thread::ThreadPool(env, "BWAPaired", num_threads_));
    request_queue_.reset(new ConcurrentQueue<std::shared_ptr<ResourceContainer<BWAReadResource>>>(capacity));
    pe_stat_queue_.reset(new ConcurrentQueue<std::shared_ptr<ResourceContainer<BWAReadResource>>>(capacity));
    finalize_queue_.reset(new ConcurrentQueue<std::shared_ptr<ResourceContainer<BWAReadResource>>>(capacity));
    init_workers();
  }

  BWAPairedExecutor::~BWAPairedExecutor() {
    if (!run_) {
      LOG(ERROR) << "Unable to safely wait in ~BWAAlignPairedOp for all threads. run_ was toggled to false\n";
    }
    while (request_queue_->size() > 0) {
      this_thread::sleep_for(chrono::milliseconds(10));
    }
    run_ = false;
    request_queue_->unblock();
    while (num_active_threads_.load(std::memory_order_relaxed) > num_threads_ - aligner_threads_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    while (pe_stat_queue_->size() > 0) {
      this_thread::sleep_for(chrono::milliseconds(10));
    }
    run_pe_stat_ = false;
    pe_stat_queue_->unblock();
    while (num_active_threads_.load(std::memory_order_relaxed) > num_threads_ - (aligner_threads_+1)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    while (finalize_queue_->size() > 0) {
      this_thread::sleep_for(chrono::milliseconds(10));
    }
    run_finalize_ = false;
    finalize_queue_->unblock();
    while (num_active_threads_.load(std::memory_order_relaxed) > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  Status BWAPairedExecutor::EnqueueChunk(std::shared_ptr<ResourceContainer<BWAReadResource> > chunk) {
    if (!compute_status_.ok()) return compute_status_;
    if (!request_queue_->push(chunk))
      return Internal("Paired executor failed to push to request queue");
    else
      return Status::OK();
  }

  void BWAPairedExecutor::init_workers() {

    auto aligner_func = [this]() {
      //std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
      int my_id = 0;
      {
        mutex_lock l(mu_);
        my_id = id_.fetch_add(1, memory_order_relaxed);
      }
      bwa_wrapper::BWAAligner aligner(options_, index_, max_read_size_);

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
        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &interval);
        vector<mem_alnreg_v>& regs = reads->get_regs();
        while (io_chunk_status.ok()) {
          //LOG(INFO) << "aligner thread " << my_id << " got interval " << interval;


          Status s = aligner.AlignSubchunk(subchunk_resource, interval, regs);

          if (!s.ok()){
            compute_status_ = s;
            return;
          }

          reads->decrement_outstanding();

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &interval);
        }
        
        if (!IsResourceExhausted(io_chunk_status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError for I/O Chunk! : " << io_chunk_status << "\n";
          compute_status_ = io_chunk_status;
          return;
        }

        if (request_queue_->drop_if_equal(reads_container))
          finalize_queue_->push(reads_container);

      }

      //VLOG(INFO) << "base aligner thread ending.";
      num_active_threads_--;
    };
    
    auto pe_stat_func = [this]() {
      int my_id = 0;
      {
        mutex_lock l(mu_);
        my_id = id_.fetch_add(1, memory_order_relaxed);
      }
      
      shared_ptr<ResourceContainer<BWAReadResource>> reads_container;
      while (run_pe_stat_) {
        // reads must be in this scope for the custom releaser to work!
        if (!pe_stat_queue_->pop(reads_container)) {
          continue;
        }

        auto *bwareads = reads_container->get();
        LOG(INFO) << "pe stat waiting for ready";
        bwareads->wait_for_ready();
        LOG(INFO) << "pe stat got ready";
        std::vector<mem_alnreg_v>& regs = bwareads->get_regs();
        mem_pestat_t* pes = bwareads->get_pes();
        LOG(INFO) << "pestat op calculating over " << regs.size() << " regs.";
        // set the pestat
        mem_pestat(options_, index_->bns->l_pac, regs.size(), &regs[0], pes);

        finalize_queue_->push(reads_container);
      }
      
      num_active_threads_--;
    };
    
    auto finalize_func = [this]() {
      int my_id = 0;
      {
        mutex_lock l(mu_);
        my_id = id_.fetch_add(1, memory_order_relaxed);
      }
      
      shared_ptr<ResourceContainer<BWAReadResource>> reads_container;
      bwa_wrapper::BWAAligner aligner(options_, index_, max_read_size_);
      Status io_chunk_status, subchunk_status;
      vector<AlignmentResultBuilder> result_builders;
      result_builders.reserve(max_secondary_+1);

      vector<BufferPair*> result_bufs;
      result_bufs.reserve(max_secondary_+1);
      ReadResource* subchunk_resource = nullptr;

      while (run_finalize_) {
        // reads must be in this scope for the custom releaser to work!
        if (!finalize_queue_->pop(reads_container)) {
          continue;
        }
        
        auto *reads = reads_container->get();

        size_t interval;
        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs, &interval);
        vector<mem_alnreg_v>& regs = reads->get_regs();
        mem_pestat_t* pes = reads->get_pes();
        while (io_chunk_status.ok()) {
          //LOG(INFO) << "finalizer thread " << my_id << " got  interval: " << interval;


          for (int i = 0; i < result_builders.size(); i++)
            result_builders[i].SetBufferPair(result_bufs[i]);

          Status s = aligner.FinalizeSubchunk(subchunk_resource, interval, regs, pes,
              result_builders);

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

      num_active_threads_--;
    };


    workers_->Schedule(pe_stat_func);
    int leftover_threads = num_threads_-1;
    aligner_threads_ = leftover_threads*thread_ratio_ < 1.0f ? 
      1 : int(leftover_threads*thread_ratio_);
    finalizer_threads_ = leftover_threads - aligner_threads_;
    LOG(INFO) << "BWA Executor init using 1 thread for PESTAT, " << aligner_threads_ 
      << " threads to ALIGN and " << finalizer_threads_ << " to FINALIZE";
    for (int i = 0; i < aligner_threads_; i++)
      workers_->Schedule(aligner_func);
    for (int i = 0; i < finalizer_threads_; i++)
      workers_->Schedule(finalize_func);
    num_active_threads_ = num_threads_;
  }

}
