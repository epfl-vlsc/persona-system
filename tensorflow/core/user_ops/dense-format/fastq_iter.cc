#include "fastq_iter.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  FASTQIterator::FASTQIterator(ResourceContainer<Data> *fastq_file) : fastq_file_(fastq_file), data_size_(fastq_file->get()->size()), data_(fastq_file->get()) {}

  Status FASTQIterator::get_next_record(const char **bases, size_t *bases_length,
                                        const char **qualities, size_t *qualities_length,
                                        const char **metadata, size_t *metadata_length)
  {
    if (!fastq_file_) {
      return Internal("get_next_record called with null data!");
    }

    TF_RETURN_IF_ERROR(read_line(metadata, metadata_length, 1)); // +1 to skip '@'
    TF_RETURN_IF_ERROR(read_line(bases, bases_length));
    TF_RETURN_IF_ERROR(skip_line());
    TF_RETURN_IF_ERROR(read_line(qualities, qualities_length));

    return Status::OK();
  }

  Status FASTQIterator::read_line(const char **line_start, size_t *line_length, size_t skip_length)
  {
    if (index_ >= data_size_) {
      return ResourceExhausted("no more records in this file");
    }

    const char *start = &data_->data()[index_] + skip_length;
    index_ += skip_length;
    const char *curr = start;
    size_t record_size = 0;
    for (; *curr != '\n' && index_ < data_size_;
         record_size++, index_++, curr++);
    *line_start = start;
    *line_length = record_size;
    index_++; // to skip over the \n character
    return Status::OK();
  }

  Status FASTQIterator::skip_line()
  {
    if (index_ >= data_size_) {
      return ResourceExhausted("no more records in this file");
    }

    const char *curr = &data_->data()[index_];
    for (; *curr != '\n' && index_ < data_size_; index_++, curr++);
    index_++; // to skip over the \n character
  }

  bool FASTQIterator::reset_iter()
  {
    index_ = 0;
    return true;
  }

  bool FASTQIterator::has_qualities()
  {
    return true;
  }

  bool FASTQIterator::has_metadata()
  {
    return true;
  }

  FASTQIterator::~FASTQIterator()
  {
    if (fastq_file_)
      fastq_file_->release();
  }

} // namespace tensorflow {
