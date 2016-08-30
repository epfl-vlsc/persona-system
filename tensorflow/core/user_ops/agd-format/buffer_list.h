#pragma once
#include <cstddef>
#include <string.h>
#include <vector>
#include <memory>
#include <atomic>
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
      std::atomic_size_t outstanding_buffers_;
      mutable mutex mu_;
      mutable std::condition_variable ready_cv_;

      void decrement_outstanding();

      friend class BufferPair;
    public:
        BufferPair& get_at(size_t index);
        void resize(size_t size);
        decltype(buf_list_)& get_when_ready();
        void reset();
    };
} // namespace tensorflow {
