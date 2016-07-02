#include "alignment_result_builder.h"
#include "util.h"

namespace tensorflow {

  void AlignmentResultBuilder::AppendAlignmentResult(const format::AlignmentResult &result, const string &string_buf)
  {
    // TODO maybe figure out result automatically?
    appendSegment(reinterpret_cast<const char*>(&result), sizeof(result), records_);
    appendSegment(string_buf.data(), string_buf.size(), records_);
  }

  void AlignmentResultBuilder::AppendAlignmentResult(const SingleAlignmentResult &result)
  {
    builder_result_.convertFromSNAP(result, builder_string_);
    AppendAlignmentResult(builder_result_, builder_string_);
  }

  // TODO support adding the SnapAlignResult object directly in here, as that is what'll be output

  void AlignmentResultBuilder::AppendAndFlush(std::vector<char> &idx_buf)
  {
    appendSegment(&records_[0], records_.size(), idx_buf);
    records_.clear();
  }
} // namespace tensorflow {
