#pragma once

#include "tensorflow/core/lib/core/errors.h"

#include <vector>

namespace tensorflow {

Status decompressGZIP(const char* segment,
                      const std::size_t segment_size,
                      std::vector<char> &output);

 Status compressGZIP(const char* segment,
                     const std::size_t segment_size,
                     std::vector<char> &output);

} // namespace tensorflow
