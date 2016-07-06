#include "column_builder.h"
#include "util.h"

namespace tensorflow {

  using namespace std;

  void AlignmentResultBuilder::AppendAlignmentResult(const format::AlignmentResult &result, const string &string_buf, vector<char> &index)
  {
    appendSegment(reinterpret_cast<const char*>(&result), sizeof(result), records_);
    appendSegment(string_buf.data(), string_buf.size(), records_);
    size_t index_entry = sizeof(result) + string_buf.size();
    // TODO just assume it's a good size for now
    index.push_back(static_cast<char>(index_entry));
  }

  void AlignmentResultBuilder::AppendAlignmentResult(const SingleAlignmentResult &result, vector<char> &index)
  {
    builder_result_.convertFromSNAP(result, builder_string_);
    AppendAlignmentResult(builder_result_, builder_string_, index);
  }

  void ColumnBuilder::AppendAndFlush(vector<char> &idx_buf)
  {
    appendSegment(&records_[0], records_.size(), idx_buf);
    records_.clear();
  }

  void StringResultBuilder::AppendString(const char* record, const std::size_t record_size, std::vector<char> &index)
  {
    appendSegment(record, record_size, records_);
    index.push_back(static_cast<char>(record_size));
  }
} // namespace tensorflow {
