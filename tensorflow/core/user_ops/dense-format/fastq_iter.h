#pragma once

#include <cstdint>
#include "tensorflow/core/lib/core/status.h"
#include "read_resource.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"

namespace tensorflow {

  class FASTQIterator : public ReadResource {
  public:

    explicit FASTQIterator(ResourceContainer<Data> *fastq_file);
    FASTQIterator() = default;
    ~FASTQIterator() override;
    FASTQIterator& operator=(FASTQIterator &&other);

    Status get_next_record(const char **bases, std::size_t *bases_length,
                           const char **qualities, std::size_t *qualities_length,
                           const char **metadata, std::size_t *metadata_length) override;

    bool reset_iter() override;

    bool has_qualities() override;
    bool has_metadata() override;

    void release() override;

  private:

    Status read_line(const char **line_start, std::size_t *line_length, std::size_t skip_length = 0);
    Status skip_line();

    ResourceContainer<Data> *fastq_file_ = nullptr;
    const Data *data_ = nullptr;
    std::size_t index_ = 0, data_size_ = 0;

    FASTQIterator(const FASTQIterator &other) = delete;
    FASTQIterator& operator=(const FASTQIterator &other) = delete;
    FASTQIterator(FASTQIterator &&other) = delete;
  };
} // namespace tensorflow {
