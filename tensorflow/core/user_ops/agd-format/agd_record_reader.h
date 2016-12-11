#pragma once

#include "read_resource.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"
#include "format.h"

namespace tensorflow {

  using namespace errors;

  /*
   * A class that provides a "view" over the data in the resource container.
   * Does not take ownership of the underlying data
   */
  class AGDRecordReader {
  public:
    AGDRecordReader(ResourceContainer<Data>* resource, size_t num_records);
    AGDRecordReader(const char* resource, size_t num_records);

    void Reset();

    Status GetNextRecord(const char** data, size_t* size);
    Status PeekNextRecord(const char** data, size_t* size);

    Status GetRecordAt(size_t index, const char** data, size_t* size);

  private:
    const format::RelativeIndex *index_;
    const char *data_, *cur_data_;
    size_t num_records_;
    size_t cur_record_ = 0;
  };

}

