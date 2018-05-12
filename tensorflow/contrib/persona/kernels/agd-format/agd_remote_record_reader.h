
#pragma once

#include "read_resource.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "data.h"
#include "format.h"
#include <vector>
#include <rados/librados.hpp>
#include <rados/buffer.h>

namespace tensorflow {

  using namespace errors;
 
  /* Provides a "view" over remote data in a ceph object store.
   * does not own buffers or io_ctx. Simply uses them to get and
   * buffer data. Only supports sequential access of records.
   *
   * `buffer` should be large enough to hold a reasonable number
   * of records. It is split in 3 and used to hold the index
   * and a double buffer of records read from Ceph.
   *
   * Generally best to avoid using this class
   */
  class AGDRemoteRecordReader {
  public:
    AGDRemoteRecordReader(string filename, size_t num_records,
        char* buffer, uint64_t buffer_size, librados::IoCtx* io_ctx);

    void Reset();

    Status GetNextRecord(const char** data, size_t* size);
    Status PeekNextRecord(const char** data, size_t* size);

    Status PrefetchRecords();
    Status Initialize();

  private:

    Status ReadData(char* dest, uint64_t size);

    struct RecordBuffer {
      char* data;
      const char* cur_data;
      uint64_t size;
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
    uint64_t base_size_ = 0;

    // two buffers, a basic double buffering scheme
    RecordBuffer buf_0_;
    RecordBuffer buf_1_;
    RecordBuffer* active_buf_ = nullptr;
    uint64_t current_offset_ = 0; // current offset into ceph object
    string filename_; // ceph object full path within `io_ctx_`

    mutable mutex mu_;
    mutable condition_variable ready_cv_;
    bool init = false;
  };

}
