#pragma once

#include "buffer.h"

namespace tensorflow {
  class BufferPair {
  private:
    Buffer index_, data_;

  public:
    decltype(index_) &index();
    decltype(data_) &data();

    void reset();
  };
} // namespace tensorflow {
