#pragma once

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/Read.h"
#include "buffer_list.h"
#include "read_resource_splitter.h"
#include <vector>
#include <memory>
#include <atomic>

namespace tensorflow {

  class ReadResourceSplitter;

  class ReadResource {
  public:
    // TODO how to declare properly?
    virtual ~ReadResource();
    /*
      Iterators over the possible input read data in this resource.

      It is the subclassing class's responsibility to set unavailable fields to 0 / null respectively
     */

    virtual Status get_next_record(Read &snap_read) = 0;
    
    virtual Status get_next_record(const char** bases, size_t* bases_len,
        const char** quals) = 0;

    virtual Status get_next_record(const char** bases, size_t* bases_len,
                                   const char** quals, const char** meta,
                                   size_t* meta_len) = 0;

    virtual std::size_t num_records();

    // Resets the iterator, and returns `true` only if the iterator was successfully reset
    // Non-reset supporting iterators may return false
    virtual bool reset_iter();

    virtual void release();

    virtual Status SplitResource(std::size_t chunk_size, ReadResourceSplitter &splitter);

    // Only valid if the subclass implements subchunks
    virtual Status split(std::size_t chunk, std::vector<BufferList*>& bl);

    virtual Status get_next_subchunk(ReadResource **rr, std::vector<BufferPair*>& b);
   
    // sometimes we don't need a buffer
    //virtual Status get_next_subchunk(ReadResource **rr);
  };

  class ReadResourceReleaser
  {
  public:
    ReadResourceReleaser(ReadResource &r);
    ~ReadResourceReleaser();

  private:
    ReadResource &rr_;
  };

} // namespace tensorflow {
