#pragma once

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

  class ReadResource {
  public:
    // TODO how to declare properly?
    virtual ~ReadResource();
    /*
      Iterators over the possible input read data in this resource.

      It is the subclassing class's responsibility to set unavailable fields to 0 / null respectively
     */
    virtual Status get_next_record(const char **bases, std::size_t *bases_length,
                                   const char **qualities, std::size_t *qualities_length,
                                   const char **metadata, std::size_t *metadata_length) = 0;

    virtual Status get_next_record(const char **bases, std::size_t *bases_length,
                                   const char **qualities, std::size_t *qualities_length);

    virtual bool has_qualities();
    virtual bool has_metadata();


    // Resets the iterator, and returns `true` only if the iterator was successfully reset
    // Non-reset supporting iterators may return false
    virtual bool reset_iter();

    virtual void release();
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
