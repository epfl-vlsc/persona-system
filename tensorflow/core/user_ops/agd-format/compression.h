#pragma once

#include "buffer.h"
#include "tensorflow/core/lib/core/errors.h"

#include <vector>
#include <zlib.h>

namespace tensorflow {

  Status decompressGZIP(const char* segment,
                        const std::size_t segment_size,
                        std::vector<char> &output);

  Status decompressGZIP(const char* segment,
                        const std::size_t segment_size,
                        Buffer *output);

  Status compressGZIP(const char* segment,
                      const std::size_t segment_size,
                      std::vector<char> &output);

  class AppendingGZIPCompressor
  {
  public:
    AppendingGZIPCompressor(Buffer *output);

    ~AppendingGZIPCompressor();

    // reinitializes the stream
    Status init();

    Status appendGZIP(const char* segment,
                      const std::size_t segment_size);

    // closes the stream
    Status finish(); // somehow flush
  private:
    z_stream stream_ = {0};
    bool done_ = false;
    Buffer *output_;
  };

} // namespace tensorflow
