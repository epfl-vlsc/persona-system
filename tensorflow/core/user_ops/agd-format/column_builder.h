#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include "format.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/AlignmentResult.h"

namespace tensorflow {

class ColumnBuilder {
public:

  void AppendAndFlush(std::vector<char> &idx_buf);

protected:

  // To be used for subclasses to build up their results
  std::vector<char> records_;
};

class AlignmentResultBuilder : public ColumnBuilder {
public:
    /*
      Append the current alignment result to the internal result buffer
      result_size is needed to ensure that the result is the correct size

      The AppendAlignmentResult Methods take a character index, which is passed in from outside to
      enable the aligner kernel to pass in a buffer from a pool, to build up a result in it.

      records_ is used to build up the actual reads, which are appended into the chunk.
    */
  void AppendAlignmentResult(const SingleAlignmentResult &result, const std::string &var_string, std::vector<char> &index);
  void AppendAlignmentResult(const SingleAlignmentResult &result, const std::string &var_string, 
                             const int flag, std::vector<char> &index);

  private:
    format::AlignmentResult builder_result_;
  };

  class StringColumnBuilder : public ColumnBuilder {
  public:

    void AppendString(const char* record, const std::size_t record_size, std::vector<char> &index);
  };

  class BaseColumnBuilder : public ColumnBuilder {
  public:

    Status AppendString(const char* bases, const std::size_t base_length, std::vector<char> &index);

  private:

    std::vector<format::BinaryBases> base_scratch_;

  };

  } // namespace tensorflow
