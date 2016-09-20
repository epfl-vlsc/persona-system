#pragma once
#include <cstddef>
#include <string.h>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>
#include "buffer.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
  class BufferList;

  class BufferPair {
  private:
    Buffer index_, data_;
    BufferList *parent_ = nullptr;
    friend class BufferList;
    void set_parent(decltype(parent_) parent);

  public:
    decltype(index_) &index();
    decltype(data_) &data();

    void reset();
    void set_ready();
  };

    class BufferList {
    private:
      std::vector<BufferPair> buf_list_;
      mutable std::atomic_size_t outstanding_buffers_;
      mutable mutex mu_;
      mutable std::condition_variable ready_cv_;

      void decrement_outstanding();
      void reset_all();

      std::size_t size_ = 0;
      std::chrono::high_resolution_clock::time_point process_start_;

      friend class BufferPair;
    public:
      BufferPair& operator[](std::size_t index);
      std::size_t size() const;
      void resize(std::size_t size);
      void wait_for_ready() const;
      void reset();

      void set_start_time();
    };
} // namespace tensorflow {
