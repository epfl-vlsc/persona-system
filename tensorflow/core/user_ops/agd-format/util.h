#pragma once

#include <vector>
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

  Status copySegment(const char* segment,
                     const std::size_t segment_size,
                     std::vector<char> &output);

  Status appendSegment(const char* segment,
                       const std::size_t segment_size,
                       std::vector<char> &output, bool double_capacity=false);

  template <typename T>
    inline void safe_reserve(std::vector<T> &v, const std::size_t ideal_length, const size_t additive_increase_bonus_size = 2 * 1024 * 1024) {
    if (v.capacity() < ideal_length) {
      v.reserve(ideal_length + additive_increase_bonus_size);
    }
  }
} // namespace tensorflow {
