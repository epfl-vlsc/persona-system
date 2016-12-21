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
    InitializeIndex();
  }
    
  AGDRecordReader::AGDRecordReader(const char* resource, size_t num_records) :
    num_records_(num_records) {
    auto idx_offset = num_records * sizeof(RelativeIndex);
    auto base_data = resource;
    index_ = reinterpret_cast<const RelativeIndex*>(base_data);
    cur_data_ = data_ = base_data + idx_offset;
    InitializeIndex();
  }

  void AGDRecordReader::InitializeIndex() {
    absolute_index_.clear();
    size_t current = 0;
    for (size_t i = 0; i < num_records_; ++i) {
      absolute_index_.push_back(current);
      current += index_[i];
    }
  }

  void AGDRecordReader::Reset() {
    cur_data_ = data_;
    cur_record_ = 0;
  }

  Status AGDRecordReader::PeekNextRecord(const char** data, size_t* size) {
    if (cur_record_ < num_records_) {
      *size = (size_t) index_[cur_record_];
      *data = cur_data_;
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
    return Status::OK();
  }

  Status AGDRecordReader::GetNextRecord(const char** data, size_t* size) {
    auto s = PeekNextRecord(data, size);
    cur_data_ += index_[cur_record_];
    if (s.ok()) {
      cur_record_++;
    }
    return s;
  }

  Status AGDRecordReader::GetRecordAt(size_t index, const char** data, size_t* size) {
    if (index < num_records_) {
      *size = index_[index];
      *data = data_ + absolute_index_[index];
    } else {
      return OutOfRange("agd record random access out of range");
    }
    return Status::OK();
  }
} // namespace tensorflow {
