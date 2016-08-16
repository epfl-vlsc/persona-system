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
} // namespace tensorflow {
