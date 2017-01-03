#pragma once

#include "read_resource.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"
#include "format.h"
#include <vector>
#include <rados/librados.hpp>
#include <rados/buffer.h>

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
    std::vector<size_t> absolute_index_;

    void InitializeIndex();
  };

  /* Provides a "view" over remote data in a ceph object store.
   * does not own buffers or io_ctx. Simply uses them to get and
   * buffer data. Only supports sequential access of records.
   *
   * `buffer` should be large enough to hold a reasonable number 
   * of records. It is split in 3 and used to hold the index
   * and a double buffer of records read from Ceph.
   */
  class AGDRemoteRecordReader {
  public:
    AGDRemoteRecordReader(string filename, size_t num_records, 
        char* buffer, size_t buffer_size, librados::IoCtx* io_ctx);

    void Reset();

    Status GetNextRecord(const char** data, size_t* size);
    Status PeekNextRecord(const char** data, size_t* size);

    Status PrefetchRecords();
    Status Initialize();

  private:
  
    Status ReadData(char* dest, size_t size);

    struct RecordBuffer {
      char* data;
      const char* cur_data;
      size_t size;
      size_t recs;
    };

    RecordBuffer* OtherBuffer();

    const format::RelativeIndex *index_;
    const char *cur_data_;
    size_t num_records_;
    size_t cur_record_ = 0;
    size_t cur_record_prefetched_ = 0;
    librados::IoCtx* io_ctx_ = nullptr;
    char* base_buf_;

    // two buffers, a basic double buffering scheme
    RecordBuffer buf_0_;
    RecordBuffer buf_1_;
    size_t base_size_ = 0;
    RecordBuffer* active_buf_ = nullptr;
    size_t current_offset_ = 0; // current offset into ceph object
    string filename_; // ceph object full path within `io_ctx_`

    mutable mutex mu_;
    mutable std::condition_variable ready_cv_;
    bool init = false;
  };

}

