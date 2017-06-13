#include "column_builder.h"
#include <chrono>
#include "util.h"

namespace tensorflow {

  using namespace std;


  void AlignmentResultBuilder::AppendEmpty() {
    char val = 0;
    index_->AppendBuffer(&val, 1);
  }


  void AlignmentResultBuilder::AppendAlignmentResult(const Alignment &result)
  {
    //LOG(INFO) << "appending alignment result location: " << result.location_ << " mapq: " << result.mapq_ << " flags: "
    //<< result.flag_ << " next: " << result.next_location_ << " template_len: " << result.template_length_ << " cigar: " << var_string;
    size_t size = result.ByteSize();
    scratch_.resize(size);
    result.SerializeToArray(&scratch_[0], size);
    ColumnBuilder::AppendRecord(&scratch_[0], size);
  }

  /*void AlignmentResultBuilder::AppendAlignmentResult(const PairedAlignmentResult &result, const size_t result_idx) {
    converted_result.convertFromSNAP(result, result_idx, 0); // TODO is 0 the correct flag to write in this case?
    data_->AppendBuffer(reinterpret_cast<const char*>(&result), sizeof(result));
    size_t index_entry = sizeof(result);
    char size = static_cast<char>(index_entry);
    index_->AppendBuffer(&size, 1);
  }*/

  
  void ColumnBuilder::SetBufferPair(BufferPair* data) {
    data->reset();
    data_ = &data->data();
    index_ = &data->index();
  }

  void ColumnBuilder::AppendRecord(const char* data, const size_t size) {
    if (size > format::MAX_INDEX_SIZE)
      LOG(ERROR) << "WARNING: Appending data larger than " << UINT8_MAX << " bytes not supported by AGD.";
    if (size > 0) // could be a zero record
      data_->AppendBuffer(data, size);
    format::RelativeIndex cSize = static_cast<uint16_t>(size);
    index_->AppendBuffer(reinterpret_cast<const char *>(&cSize), sizeof(cSize));
  }

} // namespace tensorflow {
