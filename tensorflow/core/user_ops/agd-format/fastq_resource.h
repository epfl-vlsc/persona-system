#pragma once

#include <cstdint>
#include <memory>
#include "tensorflow/core/lib/core/status.h"
#include "read_resource.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"

namespace tensorflow {

  class FastqResource : public ReadResource {
  public:
    typedef ResourceContainer<Data> FileResource;

    explicit FastqResource(std::shared_ptr<FileResource> &fastq_file, const char *start_ptr, const char *end_ptr, const std::size_t max_records);
    FastqResource() = default;
    FastqResource& operator=(FastqResource &&other);

    Status get_next_record(Read &snap_read) override;

    Status get_next_record(const char** bases, size_t* bases_len, const char** quals) override;

    bool reset_iter() override;

    void release() override;

  private:

    void read_line(const char **line_start, std::size_t *line_length, std::size_t skip_length = 0);
    void skip_line();

    std::shared_ptr<FileResource> fastq_file_; // default constructor = nullptr

    const char *start_ptr_ = nullptr, *end_ptr_ = nullptr, *current_record_ = nullptr;
    std::size_t max_records_ = 0, current_record_idx_ = 0;

    // prevent unintended copying and assignment
    FastqResource(const FastqResource &other) = delete;
    FastqResource& operator=(const FastqResource &other) = delete;
    FastqResource(FastqResource &&other) = delete;
  };
} // namespace tensorflow {
