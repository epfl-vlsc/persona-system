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
    FASTQIterator& operator=(FASTQIterator &&other);

    Status get_next_record(Read &snap_read) override;
    
    Status get_next_record(const char** bases, size_t* bases_len,
        const char** quals) override;

    bool reset_iter() override;

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
