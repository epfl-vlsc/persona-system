#include <cstddef>
#include <string.h>
#include "buffer_list.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"

namespace tensorflow {

  using namespace std;

  void BufferList::resize(size_t size) {
    auto old_size = buf_list_.size();
    if (size > old_size) {
      buf_list_.resize(size);
      // for all the extra ones, we need to initialize the parents
      for (; old_size < size; ++old_size) {
        buf_list_[old_size].set_parent(this);
      }
    }
    reset_all();
    outstanding_buffers_.store(size, memory_order_relaxed);
    size_ = size;
  }

  size_t BufferList::size() const {
    return size_;
  }

  BufferPair& BufferList::operator[](size_t index) {
    if (index >= size_) {
      LOG(ERROR) << "FATAL: get_at requested index " << index << ", with only " << size_ << " elements. Real size: " << buf_list_.size();
    }
    // using at instead of operator[] because it will error here
    return buf_list_.at(index);
  }

  void BufferList::wait_for_ready() const {
    if (outstanding_buffers_.load(memory_order_relaxed) != 0) {
      mutex_lock l(mu_);
      ready_cv_.wait(l, [this]() {
          size_t a = outstanding_buffers_.load(memory_order_relaxed);
          return a == 0;
        });
    }
  }

  void BufferList::set_start_time() {
    process_start_ = chrono::high_resolution_clock::now();
  }

  void BufferList::reset_all() {
    for (auto &b : buf_list_) {
      b.reset();
    }
  }

  void BufferList::reset() {
    reset_all();
    outstanding_buffers_.store(0, memory_order_relaxed);
    size_ = 0;
  }

  void BufferList::decrement_outstanding() {
    auto previous = outstanding_buffers_.fetch_sub(1, memory_order_relaxed);
    if (previous == 1) {
      tracepoint(bioflow, chunk_aligned,
                 chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - process_start_));
      ready_cv_.notify_one();
    }
  }

  Buffer& BufferPair::index() {
    return index_;
  }

  Buffer& BufferPair::data() {
    return data_;
  }

  void BufferPair::reset() {
    index_.reset();
    data_.reset();
  }

  void BufferPair::set_ready() {
    parent_->decrement_outstanding(); // This pointer should never be null or unset!
  }

  void BufferPair::set_parent(BufferList *parent) {
    parent_ = parent;
  }

} // namespace tensorflow {
