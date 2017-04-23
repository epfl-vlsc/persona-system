#include "read_resource_splitter.h"

namespace tensorflow {
  using namespace std;
  using namespace error;

  void ReadResourceSplitter::WaitForDone() {
    mutex_lock l(mu_);
    if (outstanding_chunks_.load(memory_order_relaxed) != 0) {
      wait_for_completion_.wait(l, [this]() {
        return outstanding_chunks_.load(memory_order_relaxed) == 0;
      });
    }
  }

  ReadResourceSplitter::~ReadResourceSplitter() {
    WaitForDone();
  }

  void ReadResourceSplitter::AddSubchunk(ReadResource *rr) {
    vector<BufferPair*> pairs;
    for (auto b : buffer_lists_) {
      b->increase_size(1);
      // add the appended last element
      pairs.push_back(&(*b)[b->size()-1]);
    }
    unique_ptr<ReadResource, function<void(ReadResource*)>> a(rr, [this](ReadResource *a) {
      SubchunkDone(a);
    });
    enqueue_batch_.push_back(make_pair(move(a), move(pairs)));
  }

  ReadResourceSplitter::ReadResourceSplitter(std::vector<BufferList *> &bl) :
          buffer_lists_(bl) {
    for (auto b : bl) {
      b->reset();
    }
  }

  void ReadResourceSplitter::EnqueueAll(TaskRunner<QueueType> &runner) {
    auto sz = enqueue_batch_.size();
    outstanding_chunks_.store(sz, memory_order_relaxed);
    runner.EnqueueMany(&enqueue_batch_[0], sz);
  }
} // namespace tensorflow {
