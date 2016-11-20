#pragma once

#include <vector>
#include <atomic>

#include "read_resource.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"
#include "format.h"

namespace tensorflow {

  class AGDPairedReadSubResource;

  class AGDPairedReadResource : public PairedReadResource {
  public:
    typedef ResourceContainer<Data> DataContainer;

    // Constructors and copy/move operators
    explicit AGDPairedReadResource() = default;

    explicit AGDPairedReadResource(std::size_t num_records, DataContainer *bases, DataContainer *quals);

    AGDPairedReadResource& operator=(AGDPairedReadResource &&other);

    Status get_next_record(std::array<std::pair<const char*, std::size_t>, 2> &bases, std::array<std::pair<const char*, std::size_t>, 2> &qualities) override;

    std::size_t num_records() override;

    // Resets the iterator, and returns `true` only if the iterator was successfully reset
    // Non-reset supporting iterators may return false
    bool reset_iter() override;

    void release() override;

    Status split(std::size_t chunk, BufferList *bl) override;

    Status get_next_subchunk(PairedReadResource **rr, BufferPair **b) override;

  private:
    DataContainer *bases_ = nullptr, *quals_ = nullptr;
    const format::RelativeIndex *base_idx_ = nullptr, *qual_idx_ = nullptr;
    std::size_t num_records_ = 0, record_idx_ = 0;
    const char *base_data_, *qual_data_;

    std::vector<AGDPairedReadSubResource> sub_resources_;
    std::atomic_size_t sub_resource_index_;
    BufferList *buffer_list_ = nullptr;

    AGDPairedReadResource(const AGDPairedReadResource &other) = delete;
    AGDPairedReadResource& operator=(const AGDPairedReadResource &other) = delete;
    AGDPairedReadResource(AGDPairedReadResource &&other) = delete;

    friend class AGDPairedReadSubResource;
  };

  class AGDPairedReadSubResource : public PairedReadResource {
  public:
    Status get_next_record(std::array<std::pair<const char*, std::size_t>, 2> &bases, std::array<std::pair<const char*, std::size_t>, 2> &qualities) override;

    std::size_t num_records() override;
  private:

    AGDPairedReadSubResource(const AGDPairedReadResource &parent_resource,
                             std::size_t index_offset, std::size_t max_idx,
                             const char *base_data_offset, const char *qual_data_offset);

    const format::RelativeIndex *base_idx_, *qual_idx_;
    const char *base_data_, *base_start_, *qual_data_, *qual_start_;
    const std::size_t start_idx_, max_idx_;
    std::size_t current_idx_;

    friend class AGDPairedReadResource;
  };
} // namespace tensorflow {
