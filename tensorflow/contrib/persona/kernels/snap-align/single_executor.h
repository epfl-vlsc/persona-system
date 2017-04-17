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
                       int max_secondary, int num_threads, int capacity) : index_(index),
                                                                           options_(options),
                                                                           num_threads_(num_threads),
                                                                           capacity_(capacity) {
      genome_ = index_->getGenome();
      // create a threadpool to execute stuff
      workers_.reset(new thread::ThreadPool(env, "SnapSingle", num_threads_));
      request_queue_.reset(new ConcurrentQueue<std::shared_ptr<ResourceContainer<ReadResource>>>(capacity));
      init_workers();
    }

    ~SnapSingleExecutor() {
      if (!run_) {
        LOG(ERROR) << "Unable to safely wait in ~SnapAlignSingleOp for all threads. run_ was toggled to false\n";
      }
      run_ = false;
      request_queue_->unblock();
      while (num_active_threads_.load(std::memory_order_relaxed) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

    }

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

    inline void init_workers();

  };
}
