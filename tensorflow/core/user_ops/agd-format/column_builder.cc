#include "column_builder.h"
#include <chrono>
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
    //auto t1 = std::chrono::high_resolution_clock::now();
    result->WriteBuffer(reinterpret_cast<const char*>(&index_[0]), index_.size());
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto writebuftime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    //LOG(INFO) << "write buf time: " << writebuftime.count();
    result->AppendBuffer(data_.data(), data_.size());
    //auto t3 = std::chrono::high_resolution_clock::now();
    //auto appendbuftime = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
    //LOG(INFO) << "append buf time: " << appendbuftime.count();
    data_.reset();
    index_.clear();
  }

} // namespace tensorflow {
