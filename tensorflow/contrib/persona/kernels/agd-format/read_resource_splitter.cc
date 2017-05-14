#include "read_resource_splitter.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  void ReadResourceSplitter::WaitForDone() {
    mutex_lock l(mu_);
    if (pending_) {
      wait_for_completion_.wait(l, [this]() {
        return !pending_;
      });
    }
  }

  Status ReadResourceSplitter::EnqueueSubchunks(ReadResource **rr, size_t count) {
    std::vector<QueueType> enqueue_batch;
    for (auto b : buffer_lists_) {
      b->resize(count);
    }
    shared_ptr<ReadResourceSplitter> a(this, [this](ReadResourceSplitter *v){
      SubchunksDone();
    });

    pair_resources_.resize(count);

    auto num_columns = buffer_lists_.size();
    for (size_t subchunk_num = 0; subchunk_num < count; ++subchunk_num) {
      auto &cc = pair_resources_[subchunk_num];
      cc.clear();
      for (size_t column_num = 0; column_num < num_columns; ++column_num) {
        auto &bl = *buffer_lists_[column_num];
        cc.push_back(&bl[subchunk_num]);
      }

      enqueue_batch.push_back(make_tuple(rr[subchunk_num], &cc, a));
    }

    if (!runner_.EnqueueMany(&enqueue_batch[0], count)) {
      return Internal("ReadResourceSplitter: EnqueueMany failed");
    }

    return Status::OK();
  }

  ReadResourceSplitter::ReadResourceSplitter(std::vector<BufferList *> &bl, TaskRunner<QueueType> &runner, ContigContainer<ColumnContainer> &pair_resources) :
          buffer_lists_(bl), runner_(runner), pair_resources_(pair_resources) {
    for (auto b : bl) {
      b->reset();
    }
  }

  void ReadResourceSplitter::SubchunksDone() {
    mutex_lock l(mu_);
    pending_ = false;
    wait_for_completion_.notify_all();
  }
} // namespace tensorflow {
