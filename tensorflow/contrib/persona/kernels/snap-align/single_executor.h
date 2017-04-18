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
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/GenomeIndex.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "tensorflow/contrib/persona/kernels/snap-align/SnapAlignerWrapper.h"

#pragma once

namespace tensorflow {


  class SnapSingleExecutor {


  public:

    SnapSingleExecutor(Env *env, GenomeIndex *index, AlignerOptions *options,
                       int max_secondary, int num_threads, int capacity);
    ~SnapSingleExecutor();

    // shared ptr is assumed to have deleter that notifies caller of completion
    // should be thread safe
    Status EnqueueChunk(std::shared_ptr<ResourceContainer < ReadResource > > chunk);


  private:
    GenomeIndex *index_ = nullptr;
    AlignerOptions *options_ = nullptr;
    const Genome *genome_ = nullptr;
    volatile bool run_ = true;
    int max_secondary_;

    std::atomic_uint_fast32_t num_active_threads_, id_{0};
    mutex mu_;

    int num_threads_;
    int capacity_;

    std::unique_ptr<ConcurrentQueue < std::shared_ptr<ResourceContainer < ReadResource>>>> request_queue_;

    Status compute_status_ = Status::OK();
    std::unique_ptr<thread::ThreadPool> workers_;

    void init_workers();

  };
}