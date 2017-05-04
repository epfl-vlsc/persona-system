#pragma once

#include "read_resource.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/contrib/persona/kernels/executor/task_runner.h"
#include <atomic>
#include <condition_variable>
#include <vector>

namespace tensorflow {

  class ReadResource;

  class ReadResourceSplitter {
  public:
    typedef std::tuple<ReadResource*, std::vector<BufferPair*>, std::shared_ptr<ReadResourceSplitter>> QueueType;
    ReadResourceSplitter(std::vector<BufferList*> &bl);

    void AddSubchunks(ReadResource *rr[], std::size_t count);
    Status EnqueueAll(TaskRunner<QueueType> &runner);

    void WaitForDone();

  private:
    void SubchunksDone();

    mutable mutex mu_;
    mutable std::condition_variable wait_for_completion_;
    std::vector<BufferList*> &buffer_lists_;
    std::vector<QueueType> enqueue_batch_;
    volatile bool pending_ = true;
  };
} // namespace tensorflow {
