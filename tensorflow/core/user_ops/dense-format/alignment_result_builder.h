#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include "format.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/AlignmentResult.h"

namespace tensorflow {

class AlignmentResultBuilder {
public:
  /*
    Append the current alignment result to the internal result buffer
    result_size is needed to ensure that the result is the correct size
   */
  void AppendAlignmentResult(const format::AlignmentResult &result, const std::string &string_buf, std::vector<char> &index);

  void AppendAlignmentResult(const SingleAlignmentResult &result, std::vector<char> &index);

  void AppendAndFlush(std::vector<char> &idx_buf);

private:
  /* A buffer to build up the records. This is copy-appended to the "scratch" buffer
     in the public methods above
   */
  std::vector<char> records_;
  std::string builder_string_;
  format::AlignmentResult builder_result_;
};

} // namespace tensorflow
