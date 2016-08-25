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
  class Buffer;
    class BufferList {
    private:
      std::vector<Buffer> buf_list_;
      std::atomic_size_t outstanding_buffers_;
      mutable mutex mu_;
      mutable std::condition_variable ready_cv_;

      void decrement_outstanding();

      friend class Buffer;
    public:
        Buffer* get_at(size_t index);
        void resize(size_t size);
        decltype(buf_list_)& get_when_ready();
        void reset();
    };
} // namespace tensorflow {
