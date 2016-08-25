#include <cstddef>
#include <string.h>
#include "buffer_list.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

  using namespace std;

  void BufferList::resize(size_t size) {
    buf_list_.resize(size);
    outstanding_buffers_ = size;
  }

  vector<Buffer>& BufferList::get_when_ready() {
    if (outstanding_buffers_.load() != 0) {
      mutex_lock l(mu_);
      ready_cv_.wait(l, [this]() {
          auto a = outstanding_buffers_.load();
          return outstanding_buffers_.load() == 0;
        });
    }
    return buf_list_;
  }

  Buffer* BufferList::get_at(size_t index) {
    if (index >= buf_list_.size()) {
      resize(index+1);
      outstanding_buffers_++;
    }
    auto &a = buf_list_[index];
    a.set_buffer_list_parent(this);

    return &a;
  }

  void BufferList::reset() {
    for (auto &b : buf_list_) {
      b.reset();
    }
    outstanding_buffers_ = 0;
  }

  void BufferList::decrement_outstanding() {
    size_t previous = --outstanding_buffers_;
    if (previous == 0) {
      ready_cv_.notify_one();
    }
  }

} // namespace tensorflow {
