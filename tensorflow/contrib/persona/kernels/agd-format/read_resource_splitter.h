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
    typedef std::pair<std::unique_ptr<ReadResource, std::function<void(ReadResource*)>>, std::vector<BufferPair*>> QueueType;
    ReadResourceSplitter(std::vector<BufferList*> &bl);
    ~ReadResourceSplitter();

    void AddSubchunk(ReadResource *rr);
    void EnqueueAll(TaskRunner<QueueType> &runner);

  private:
    void SubchunkDone(ReadResource *rr);

    void WaitForDone();

    std::atomic_uint_fast16_t outstanding_chunks_{0};
    mutable mutex mu_;
    mutable std::condition_variable wait_for_completion_;
    std::vector<BufferList*> &buffer_lists_;
    std::vector<QueueType> enqueue_batch_;
  };
} // namespace tensorflow {
