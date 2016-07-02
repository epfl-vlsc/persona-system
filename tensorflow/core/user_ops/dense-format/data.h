#pragma once

#include <cstddef>

namespace tensorflow {
  class Data {
  public:
    virtual const char* data() const = 0;
    virtual std::size_t size() const = 0;
  };
} // namespace tensorflow {
