#include "column_builder.h"
#include <chrono>
#include "util.h"

namespace tensorflow {

  using namespace std;

  void AlignmentResultBuilder::AppendAlignmentResult(const SingleAlignmentResult &result, const string &var_string, const int flag)
  {
    converted_result.convertFromSNAP(result, flag);
    data_->AppendBuffer(reinterpret_cast<const char*>(&converted_result), sizeof(converted_result));
    data_->AppendBuffer(var_string.data(), var_string.size());
    size_t index_entry = sizeof(converted_result) + var_string.size();
    char size = static_cast<char>(index_entry);
    index_->AppendBuffer(&size, 1);
  }

  void AlignmentResultBuilder::AppendAlignmentResult(const SingleAlignmentResult &result) {
    data_->AppendBuffer(reinterpret_cast<const char*>(&result), sizeof(result));
    size_t index_entry = sizeof(result);
    char size = static_cast<char>(index_entry);
    index_->AppendBuffer(&size, 1);
  }

  void AlignmentResultBuilder::set_buffer_pair(BufferPair *data) {
    data->reset();
    data_ = &data->data();
    index_ = &data->index();
  }

  void AlignmentResultBuilder::AppendAlignmentResult(const PairedAlignmentResult &result, const size_t result_idx, const string &var_string, const int flag)
  {
    converted_result.convertFromSNAP(result, result_idx, flag);
    data_->AppendBuffer(reinterpret_cast<const char*>(&converted_result), sizeof(converted_result));
    data_->AppendBuffer(var_string.data(), var_string.size());
    size_t index_entry = sizeof(converted_result) + var_string.size();
    char size = static_cast<char>(index_entry);
    index_->AppendBuffer(&size, 1);
  }

  void AlignmentResultBuilder::AppendAlignmentResult(const PairedAlignmentResult &result, const size_t result_idx) {
    converted_result.convertFromSNAP(result, result_idx, 0); // TODO is 0 the correct flag to write in this case?
    data_->AppendBuffer(reinterpret_cast<const char*>(&result), sizeof(result));
    size_t index_entry = sizeof(result);
    char size = static_cast<char>(index_entry);
    index_->AppendBuffer(&size, 1);
  }

  
  void ColumnBuilder::SetBufferPair(BufferPair* data) {
    data->reset();
    data_ = &data->data();
    index_ = &data->index();
  }

  void ColumnBuilder::AppendRecord(const char* data, const std::size_t size) {
    if (size > 255) LOG(INFO) << "WARNING: Appending data larger than 255 bytes not supported by AGD.";
    data_->AppendBuffer(data, size);
    char cSize = static_cast<char>(size);
    index_->AppendBuffer(&cSize, 1);
  }

} // namespace tensorflow {
