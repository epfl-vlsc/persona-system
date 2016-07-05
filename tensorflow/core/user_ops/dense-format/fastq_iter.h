#pragma once

#include <cstdint>
#include "tensorflow/core/lib/core/status.h"
#include "data.h"

namespace tensorflow {

  class FASTQIterator {
  public:

    FASTQIterator(const Data *data);
    FASTQIterator();

    Status get_next_record(const char **bases, std::size_t *bases_length,
                           const char **qualities, std::size_t *qualities_length,
                           const char **metadata, std::size_t *metadata_length);

    // TODO implement if we need to iterate multiple times
    // currently not needed
    //void reset_iter();

  private:

    Status read_line(const char **line_start, std::size_t *line_length);
    Status skip_line();

    const Data *data_;
    size_t index_ = 0, data_size_ = 0;
  };
} // namespace tensorflow {
