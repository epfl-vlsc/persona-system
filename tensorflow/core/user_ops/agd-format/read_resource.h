#pragma once

#include "tensorflow/core/lib/core/status.h"
#include "buffer_list.h"
#include <array>
#include <utility>
#include <memory>
#include <atomic>

namespace tensorflow {

  class ReleasableRead {
  public:
    virtual void release() = 0;
  };

  class ReadResource : public ReleasableRead {
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

    virtual std::size_t num_records();

    // Resets the iterator, and returns `true` only if the iterator was successfully reset
    // Non-reset supporting iterators may return false
    virtual bool reset_iter();

    virtual void release() override;

    // Only valid if the subclass implements subchunks
    virtual Status split(std::size_t chunk, BufferList *bl);

    virtual Status get_next_subchunk(ReadResource **rr, BufferPair **b);
  };

  class ReadResourceReleaser
  {
  public:
    ReadResourceReleaser(ReleasableRead &r);
    ~ReadResourceReleaser();

  private:
    ReleasableRead &rr_;
  };

  class PairedReadResource : public ReleasableRead {
  public:
    virtual Status get_next_record(std::array<std::pair<const char*, std::size_t>, 2> &bases, std::array<std::pair<const char*, std::size_t>, 2> &qualities) = 0;

    virtual std::size_t num_records();

    // Resets the iterator, and returns `true` only if the iterator was successfully reset
    // Non-reset supporting iterators may return false
    virtual bool reset_iter();

    virtual void release() override;

    // Only valid if the subclass implements subchunks
    virtual Status split(std::size_t chunk, BufferList *bl);

    virtual Status get_next_subchunk(PairedReadResource **rr, BufferPair **b);
  };

} // namespace tensorflow {
