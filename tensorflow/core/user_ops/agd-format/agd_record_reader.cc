#include "agd_record_reader.h"

namespace tensorflow {

  using namespace errors;
  using namespace std;
  using namespace format;

  // TODO we should probably do some more checks on this
  AGDRecordReader::AGDRecordReader(ResourceContainer<Data>* resource, size_t num_records) :
    num_records_(num_records) {
    auto idx_offset = num_records * sizeof(RelativeIndex);
    auto base_data = resource->get()->data();
    index_ = reinterpret_cast<const RelativeIndex*>(base_data);
    cur_data_ = data_ = base_data + idx_offset;
  }

  void AGDRecordReader::Reset() {
    cur_data_ = data_;
    cur_record_ = 0;
  }

  Status AGDRecordReader::GetNextRecord(const char** data, size_t* size) {
    if (cur_record_ < num_records_) {
      *size = (size_t) index_[cur_record_];
      *data = cur_data_;
      cur_data_ += index_[cur_record_++];
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
    return Status::OK();
  }

  Status AGDRecordReader::GetRecordAt(size_t index, const char** data, size_t* size) {
    if (index < num_records_) {
      size_t offset = 0;
      *size = index_[0];
      decltype(index) i;

      for (i = 0; i < index; i++) {
        offset += index_[i];
      }

      *size = index_[i-1];
      *data = data_ + offset;
    } else {
      return OutOfRange("agd record random access out of range");
    }
    return Status::OK();
  }
} // namespace tensorflow {
