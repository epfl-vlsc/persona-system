#pragma once

#include "read_resource.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"
#include "format.h"

namespace tensorflow {
  class DenseReadResource : public ReadResource {
  public:
    typedef ResourceContainer<Data> DataContainer;

    /*
      Note that this iterator assumes that the Data in each of the possible containers has already been verified in a prior step
     */

    explicit DenseReadResource() = default;

    explicit DenseReadResource(std::size_t num_records, DataContainer *bases, DataContainer *quals, DataContainer *meta);
    ~DenseReadResource() override;

    Status get_next_record(const char **bases, std::size_t *bases_length,
                           const char **qualities, std::size_t *qualities_length,
                           const char **metadata, std::size_t *metadata_length) override;

    bool reset_iter() override;

    bool has_qualities() override;
    bool has_metadata() override;

    void release() override;

  private:
    DataContainer *bases_, *quals_, *meta_;
    const format::RecordTable *base_idx_, *qual_idx_, *meta_idx_;
    const char *base_data_, *qual_data_, *meta_data_;
    std::size_t num_records_, record_idx_ = 0;
  };
} // namespace tensorflow {
