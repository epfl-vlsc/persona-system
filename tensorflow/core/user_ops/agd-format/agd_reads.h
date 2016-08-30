#pragma once

#include "read_resource.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"
#include "format.h"
#include <vector>
#include <atomic>

namespace tensorflow {

  class AGDReadSubResource;

  class AGDReadResource : public ReadResource {
  public:
    typedef ResourceContainer<Data> DataContainer;

    /*
      Note that this iterator assumes that the Data in each of the possible containers has already been verified in a prior step
     */

    explicit AGDReadResource() = default;

    explicit AGDReadResource(std::size_t num_records, DataContainer *bases, DataContainer *quals, DataContainer *meta);
    explicit AGDReadResource(std::size_t num_records, DataContainer *bases, DataContainer *quals);

    AGDReadResource& operator=(AGDReadResource &&other);

    // WARNING: this method assumes that all fields are populated (qual, base, meta)
    // If this isn't the case but you call this method, segfault / undefinied behavior is likely
    // this avoids unnecessary conditionals
    Status get_next_record(const char **bases, std::size_t *bases_length,
                           const char **qualities, std::size_t *qualities_length,
                           const char **metadata, std::size_t *metadata_length) override;

    Status get_next_record(const char **bases, std::size_t *bases_length,
                           const char **qualities, std::size_t *qualities_length) override;

    bool reset_iter() override;

    bool has_qualities() override;
    bool has_metadata() override;

    void release() override;

    std::size_t num_records() override;

    Status split(std::size_t chunk, BufferList *bl) override;
    Status get_next_subchunk(ReadResource **rr, BufferPair **b) override;

  private:
    DataContainer *bases_ = nullptr, *quals_ = nullptr, *meta_ = nullptr;
    const format::RecordTable *base_idx_ = nullptr, *qual_idx_ = nullptr, *meta_idx_ = nullptr;
    const char *base_data_ = nullptr, *qual_data_ = nullptr, *meta_data_ = nullptr;
    std::size_t num_records_ = 0, record_idx_ = 0;

    AGDReadResource(const AGDReadResource &other) = delete;
    AGDReadResource& operator=(const AGDReadResource &other) = delete;
    AGDReadResource(AGDReadResource &&other) = delete;
    std::vector<AGDReadSubResource> sub_resources_;
    std::atomic_size_t sub_resource_index_;
    BufferList *buffer_list_ = nullptr;

    friend class AGDReadSubResource;
  };

  class AGDReadSubResource : public ReadResource {
    friend class AGDReadResource;

  private:

    AGDReadSubResource(const AGDReadResource &parent_resource,
                         std::size_t index_offset, std::size_t max_idx,
                         const char *base_data_offset, const char *qual_data_offset, const char *meta_data_offset);

  public:
    Status get_next_record(const char **bases, std::size_t *bases_length,
                           const char **qualities, std::size_t *qualities_length,
                           const char **metadata, std::size_t *metadata_length) override;

    Status get_next_record(const char **bases, std::size_t *bases_length,
                           const char **qualities, std::size_t *qualities_length) override;

    bool reset_iter() override;
    bool has_qualities() override;
    bool has_metadata() override;

    std::size_t num_records() override;

  private:
    const format::RecordTable *base_idx_, *qual_idx_, *meta_idx_;
    const char *base_data_, *base_start_, *qual_data_, *qual_start_, *meta_data_, *meta_start_;
    const std::size_t start_idx_, max_idx_;
    std::size_t current_idx_;
  };
} // namespace tensorflow {
