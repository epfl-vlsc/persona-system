#include "column_builder.h"
#include "util.h"

namespace tensorflow {

  using namespace std;

  void AlignmentResultBuilder::AppendAlignmentResult(const SingleAlignmentResult &result, const std::string &var_string, const int flag)
  {
    format::AlignmentResult converted_result;
    converted_result.convertFromSNAP(result, flag);
    data_.AppendBuffer(reinterpret_cast<const char*>(&converted_result), sizeof(converted_result));
    data_.AppendBuffer(var_string.data(), var_string.size());
    size_t index_entry = sizeof(converted_result) + var_string.size();
    index_.push_back(static_cast<decltype(index_)::value_type>(index_entry));
  }

  void AlignmentResultBuilder::WriteResult(Buffer *result)
  {
    result->WriteBuffer(reinterpret_cast<const char*>(&index_[0]), index_.size());
    result->AppendBuffer(data_.data(), data_.size());
    data_.reset();
    index_.clear();
  }

} // namespace tensorflow {
