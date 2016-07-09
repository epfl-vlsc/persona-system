#pragma once

#include "read_resource.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"

namespace tensorflow {
  class DenseReadResource : public ReadResource {
  public:
    typedef ResourceContainer<Data> DataContainer;

    explicit DenseReadResource(DataContainer *bases, DataContainer *quals, DataContainer *meta);
    ~DenseReadResource() override;

    Status get_next_record(const char **bases, std::size_t *bases_length,
                           const char **qualities, std::size_t *qualities_length,
                           const char **metadata, std::size_t *metadata_length) override;

    bool reset_iter() override;

    bool has_qualities() override;
    bool has_metadata() override;

  private:
    DataContainer *bases_, *quals_, *meta_;
  };
} // namespace tensorflow {
