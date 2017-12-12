#pragma once

#include <cstdint>
#include <memory>
#include <atomic>
#include "tensorflow/core/lib/core/status.h"
#include "read_resource.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "data.h"

namespace tensorflow {

  class FastaResource : public ReadResource {
  public:
    typedef ResourceContainer<Data> FileResource;

    explicit FastaResource(std::shared_ptr<FileResource> &fasta_file,
                           std::shared_ptr<std::atomic<unsigned int>> &use_count,
                           std::shared_ptr<volatile bool> &done,
                           const char *start_ptr, const char *end_ptr, const std::size_t max_records);
    FastaResource() = default;
    FastaResource& operator=(FastaResource &&other);

    Status get_next_record(Read &snap_read) override;

    // fasta has no quals, so quals is always set to nullptr here
    // bases may contain newlines (depends on the particular file)
    // downstream needs to remove these
    Status get_next_record(const char** bases, size_t* bases_len, const char** quals) override;

    // same as above
    Status get_next_record(const char** bases, size_t* bases_len, const char** quals,
                           const char** meta, size_t* meta_len) override;

    bool reset_iter() override;

    void release() override;

    std::size_t num_records();

  private:

    void read_line(const char **line_start, std::size_t *line_length, std::size_t skip_length = 0);

    std::shared_ptr<FileResource> fasta_file_; // default constructor = nullptr

    // We must use a shared separate atomic because we need to release the file when all are done
    // and we can't rely on the custom destructor trick because FileResource is, like this type
    // a shared resouce, and we can't rely on any ordering of resource deletion
    // when the session is destroyed
    std::shared_ptr<std::atomic<unsigned int>> file_use_count_;
    std::shared_ptr<volatile bool> done_;

    const char *start_ptr_ = nullptr, *end_ptr_ = nullptr, *current_record_ = nullptr;
    std::size_t max_records_ = 0, current_record_idx_ = 0;

    // prevent unintended copying and assignment
    FastaResource(const FastaResource &other) = delete;
    FastaResource& operator=(const FastaResource &other) = delete;
    FastaResource(FastaResource &&other) = delete;
  };
} // namespace tensorflow {
