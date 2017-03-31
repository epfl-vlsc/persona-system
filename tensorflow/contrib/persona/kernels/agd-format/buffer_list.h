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

  public:
    decltype(index_) &index();
    decltype(data_) &data();

    void reset();
  };

    class BufferList {
    private:
      std::vector<BufferPair> buf_list_;

      void reset_all();

      std::size_t size_ = 0;

    public:
      BufferPair& operator[](std::size_t index);
      std::size_t size() const;
      void resize(std::size_t size);
      void reset();
    };
} // namespace tensorflow {
