#include "fastq_iter.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  FASTQIterator::FASTQIterator(const Data *data) : data_(data) {}
  FASTQIterator::FASTQIterator() : data_(nullptr) {}

  Status FASTQIterator::get_next_record(const char **bases, size_t *bases_length,
                                        const char **qualities, size_t *qualities_length,
                                        const char **metadata, size_t *metadata_length)
  {
    if (!data_) {
      // TODO return an error
    }

    // attempt to parse from the current record, adding length every way
    return Status::OK();
  }

} // namespace tensorflow {
