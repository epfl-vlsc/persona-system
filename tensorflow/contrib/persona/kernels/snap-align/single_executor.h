//
// Created by Stuart Byma on 17/04/17.
//

#pragma once

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
#include "tensorflow/contrib/persona/kernels/executor/task_runner.h"


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

  class SnapSingle : public TaskRunner<std::unique_ptr<ReadResource>> {
  public:
    SnapSingle(Env *env, GenomeIndex *index, AlignerOptions *options,
               uint16_t max_secondary, uint16_t num_threads);

    Status Start() override;
  private:
    GenomeIndex *index_;
    AlignerOptions *options_;
    uint16_t max_secondary_, num_threads_;
  };
}
