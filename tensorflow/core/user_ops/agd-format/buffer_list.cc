#include <cstddef>
#include <string.h>
#include "buffer_list.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

  using namespace std;

  void BufferList::resize(size_t size) {
    buf_list_.resize(size);
    outstanding_buffers_.store(size, memory_order_relaxed);
  }

  vector<Buffer>& BufferList::get_when_ready() {
    if (outstanding_buffers_.load(memory_order_relaxed) != 0) {
      mutex_lock l(mu_);
      ready_cv_.wait(l, [this]() {
          return outstanding_buffers_.load(memory_order_relaxed) == 0;
        });
    }
    return buf_list_;
  }

  Buffer* BufferList::get_at(size_t index) {
    if (index >= buf_list_.size()) {
      resize(index+1);
      outstanding_buffers_.fetch_add(1, memory_order_relaxed);
    }
    auto &a = buf_list_[index];
    a.set_buffer_list_parent(this);

    return &a;
  }

  void BufferList::reset() {
    for (auto &b : buf_list_) {
      b.reset();
    }
    outstanding_buffers_.store(0, memory_order_relaxed);
  }

  void BufferList::decrement_outstanding() {
    auto previous = outstanding_buffers_.fetch_sub(1, memory_order_relaxed);
    if (previous == 1) {
      ready_cv_.notify_one();
    }
  }

} // namespace tensorflow {
