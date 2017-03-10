#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include "format.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/AlignmentResult.h"
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

  //void AppendAlignmentResult(const SingleAlignmentResult &result, const std::string &var_string, const int flag);

  //void AppendAlignmentResult(const SingleAlignmentResult &result);

  // This is the only one we should use now
  void AppendAlignmentResult(const format::AlignmentResult &result, const string &var_string);

  // sometimes we want to append an empty result 
  // e.g. not all reads will generate X secondary alignments (so some columns will have gaps)
  void AppendEmpty();

  //void AppendAlignmentResult(const PairedAlignmentResult &result, const std::size_t result_idx);

  private:
    Buffer *data_ = nullptr, *index_ = nullptr;
    format::AlignmentResult converted_result;
  };

class ColumnBuilder {
  public:
    void SetBufferPair(BufferPair* data);

    void AppendRecord(const char* data, const std::size_t size);

  private:
    Buffer *data_ = nullptr, *index_ = nullptr;
};

} // namespace tensorflow
