#include "read_resource_splitter.h"

namespace tensorflow {
  using namespace std;
  using namespace error;

  void ReadResourceSplitter::WaitForDone() {
    mutex_lock l(mu_);
    if (pending_) {
      wait_for_completion_.wait(l, [this]() {
        return !pending_;
      });
    }
  }

  ReadResourceSplitter::~ReadResourceSplitter() {
    WaitForDone();
  }

  void ReadResourceSplitter::AddSubchunks(ReadResource *rr[], size_t count) {
    vector<BufferPair*> pairs;
    for (auto b : buffer_lists_) {
      b->resize(count);
    }
    shared_ptr<ReadResourceSplitter> a(this, [this](ReadResourceSplitter *v){
      SubchunksDone();
    });

    auto num_columns = buffer_lists_.size();
    for (size_t subchunk_num = 0; subchunk_num < count; ++subchunk_num) {
      vector <BufferPair*> bps;
      bps.reserve(num_columns);
      for (size_t column_num = 0; column_num < num_columns; ++column_num) {
        bps[column_num] = &(*buffer_lists_[column_num])[subchunk_num];
      }

      enqueue_batch_.push_back(make_tuple(rr[subchunk_num], move(bps), a));
    }
  }

  ReadResourceSplitter::ReadResourceSplitter(std::vector<BufferList *> &bl) :
          buffer_lists_(bl) {
    for (auto b : bl) {
      b->reset();
    }
  }

  void ReadResourceSplitter::EnqueueAll(TaskRunner<QueueType> &runner) {
    auto sz = enqueue_batch_.size();
    runner.EnqueueMany(&enqueue_batch_[0], sz);
  }

  void ReadResourceSplitter::SubchunksDone() {
    mutex_lock l(mu_);
    pending_ = false;
    wait_for_completion_.notify_all();
  }
} // namespace tensorflow {