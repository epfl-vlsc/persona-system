#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include "format.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/AlignmentResult.h"
#include "buffer_list.h"

namespace tensorflow {

class AlignmentResultBuilder {
public:
    /*
      Append the current alignment result to the internal result buffer
      result_size is needed to ensure that the result is the correct size

      The AppendAlignmentResult Methods take a character index, which is passed in from outside to
      enable the aligner kernel to pass in a buffer from a pool, to build up a result in it.

      records_ is used to build up the actual reads, which are appended into the chunk.
    */
  void set_buffer_pair(BufferPair *data);

  void AppendAlignmentResult(const SingleAlignmentResult &result, const std::string &var_string, const int flag);

  void AppendAlignmentResult(const SingleAlignmentResult &result);

  private:
    Buffer *data_ = nullptr, *index_ = nullptr;
    format::AlignmentResult converted_result;
  };

} // namespace tensorflow
