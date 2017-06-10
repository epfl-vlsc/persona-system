//
// Created by Stuart Byma on 17/04/17.
//

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>
#include <utility>
#include <chrono>
#include <atomic>
#include <vector>
#include <thread>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/bwa-align/bwa_wrapper.h"
#include "tensorflow/contrib/persona/kernels/bwa-align/bwa_reads.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"

#pragma once

namespace tensorflow {


  class BWAPairedExecutor {


  public:

    BWAPairedExecutor(Env *env, bwaidx_t *index, mem_opt_t *options,
                      int max_secondary, int num_threads, int capacity,
                      int max_read_size, float thread_ratio);
    ~BWAPairedExecutor();

    // shared ptr is assumed to have deleter that notifies caller of completion
    // should be thread safe
    Status EnqueueChunk(std::shared_ptr<ResourceContainer < BWAReadResource > > chunk);


  private:
    volatile bool run_ = true, run_pe_stat_ = true, run_finalize_ = true;
    int max_secondary_, max_read_size_;
    float thread_ratio_ = 0.66f;
    
    bwaidx_t* index_ = nullptr;
    mem_opt_t* options_ = nullptr;

    std::atomic_uint_fast32_t num_active_threads_, id_{0};
    mutex mu_;

    int num_threads_;
    int aligner_threads_;
    int finalizer_threads_;
    int capacity_;

    std::unique_ptr<ConcurrentQueue < std::shared_ptr<ResourceContainer < BWAReadResource>>>> request_queue_;
   
    std::unique_ptr<ConcurrentQueue < std::shared_ptr<ResourceContainer < BWAReadResource>>>> pe_stat_queue_;
    
    std::unique_ptr<ConcurrentQueue < std::shared_ptr<ResourceContainer < BWAReadResource>>>> finalize_queue_;

    Status compute_status_ = Status::OK();
    std::unique_ptr<thread::ThreadPool> workers_;

    void init_workers();

  };
}
