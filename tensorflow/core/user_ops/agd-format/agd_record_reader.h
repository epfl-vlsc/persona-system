#pragma once

#include "read_resource.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"
#include "format.h"

namespace tensorflow {

  using namespace errors;

  class AGDRecordReader {
    public:
      AGDRecordReader(ResourceContainer<Data>* resource, int num_records) {
        num_records_ = num_records;
        auto idx_offset = num_records * sizeof(RelativeIndex);
        base_resource_ = resource;
        base_data_ = base_resource_->get()->data();
        index_ = reinterpret_cast<const RelativeIndex*>(base_data_);
        cur_data_ = data_ = base_data_ + idx_offset;
        cur_record_ = 0;
      }

      void Reset() {
        cur_data_ = data_;
        cur_record_ = 0;
      }

      Status GetNextRecord(const char** data, uint32* size) {
        if (cur_record_ < num_records_) {
          *size = (uint32) index_[cur_record_];
          *data = cur_data_;
          cur_data_ += index_[cur_record_];
          cur_record_++;
          return Status::OK();
        } else
          return ResourceExhausted("agd record container exhausted");
      }

      Status GetRecordAt(const char** data, uint32* size, int index) {
        if (index < num_records) {
          uint32 offset = 0;
          *size = index_[0];
          for (int i = 0; i < index; i++) {
            offset += index_[i];
            *size = index_[i];
          }
          *data = data_ + offset;
        } else
          return OutOfRange("agd record random access out of range");
      }

    private:
      const format::RelativeIndex *index_, cur_index_;
      const char *base_data_, data_, cur_data_;
      int num_records_;
      int cur_record_;
  }

}

