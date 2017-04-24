#pragma once

#include "buffer_pair.h"

namespace tensorflow {
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
