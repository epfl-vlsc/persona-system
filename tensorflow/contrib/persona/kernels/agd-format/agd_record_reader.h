#pragma once

#include "read_resource.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "data.h"
#include "format.h"
#include <vector>

namespace tensorflow {

  using namespace errors;

  /*
   * A class that provides a "view" over the data in the resource container.
   * Does not take ownership of the underlying data
   *
   * This is the class to inherit from if you want another interface to
   * AGD chunk records.
   */
  class AGDRecordReader {
  public:
    AGDRecordReader(ResourceContainer<Data>* resource, size_t num_records);
    AGDRecordReader(const char* resource, size_t num_records);

    // This constructor is yet more technical debt we'll have to pay back eventually
    // It is used for the merge op, and assumes that it gets data with a header
    static AGDRecordReader fromUncompressed(ResourceContainer<Data> *resource, bool *success);

    void Reset();
    int NumRecords() { return num_records_; }

    Status GetNextRecord(const char** data, size_t* size);
    Status PeekNextRecord(const char** data, size_t* size);

    Status GetRecordAt(size_t index, const char** data, size_t* size);

    size_t GetCurrentIndex() {
      return cur_record_;
    }

  private:
    const format::RelativeIndex *index_;
    const char *data_, *cur_data_;
    size_t cur_record_ = 0;
    std::vector<size_t> absolute_index_;

    void InitializeIndex();

  protected:
    size_t num_records_;

  };


}
