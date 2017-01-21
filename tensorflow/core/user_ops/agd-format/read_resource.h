#pragma once

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/Read.h"
#include "buffer_list.h"
#include <vector>
#include <memory>
#include <atomic>

namespace tensorflow {

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

    virtual std::size_t num_records();

    // Resets the iterator, and returns `true` only if the iterator was successfully reset
    // Non-reset supporting iterators may return false
    virtual bool reset_iter();

    virtual void release();

    // Only valid if the subclass implements subchunks
    virtual Status split(std::size_t chunk, BufferList *bl);

    virtual Status get_next_subchunk(ReadResource **rr, BufferPair **b);
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
