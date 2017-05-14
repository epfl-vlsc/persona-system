#pragma once

#include "read_resource.h"
#include "contig_container.h"
#include "buffer_pair.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/contrib/persona/kernels/executor/task_runner.h"
#include <atomic>
#include <condition_variable>
#include <vector>

namespace tensorflow {

  class ReadResource;

  class ReadResourceSplitter {
  public:
    typedef ContigContainer<BufferPair*> ColumnContainer;
    typedef std::tuple<ReadResource*, ColumnContainer*, std::shared_ptr<ReadResourceSplitter>> QueueType;

    ReadResourceSplitter(std::vector<BufferList*> &bl, TaskRunner<QueueType> &runner,
                         ContigContainer<ColumnContainer> &pair_resources);

    Status EnqueueSubchunks(ReadResource **rr, std::size_t count);

    void WaitForDone();

  private:
    void SubchunksDone();

    mutable mutex mu_;
    mutable std::condition_variable wait_for_completion_;
    std::vector<BufferList*> &buffer_lists_;
    ContigContainer<ColumnContainer> &pair_resources_;
    TaskRunner<QueueType> &runner_;
    volatile bool pending_ = true;
  };
} // namespace tensorflow {
