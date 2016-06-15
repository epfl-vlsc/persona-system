#ifndef TENSORFLOW_CORE_USEROPS_DENSE_FORMAT_DECOMPRESS_H_
#define TENSORFLOW_CORE_USEROPS_DENSE_FORMAT_DECOMPRESS_H_

#include "tensorflow/core/lib/core/errors.h"

#include <vector>

namespace tensorflow {

Status decompressGZIP(const char* segment,
                      const std::size_t segment_size,
                      std::vector<char> &output);

} // namespace tensorflow

#endif
