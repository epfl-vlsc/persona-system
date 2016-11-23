#include "fastq_iter.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  FASTQIterator& FASTQIterator::operator=(FASTQIterator &&other) {
    fastq_file_ = other.fastq_file_;
    other.fastq_file_ = nullptr;
    data_ = other.data_;
    other.data_ = nullptr;

    // No need to reset these other things
    index_ = other.index_;
    data_size_ = other.data_size_;
    return *this;
  }

  FASTQIterator::FASTQIterator(ResourceContainer<Data> *fastq_file) : fastq_file_(fastq_file), data_size_(fastq_file->get()->size()), data_(fastq_file->get()) {}

  Status FASTQIterator::get_next_record(Read &snap_read)
  {
    if (!fastq_file_) {
      return Internal("get_next_record called with null data!");
    }
    const char *meta, *base, *qual;
    size_t meta_len, base_len;

    TF_RETURN_IF_ERROR(read_line(&meta, &meta_len, 1)); // +1 to skip '@'
    TF_RETURN_IF_ERROR(read_line(&base, &base_len));
    TF_RETURN_IF_ERROR(skip_line());
    TF_RETURN_IF_ERROR(read_line(&qual, &base_len));
    snap_read.init(meta, meta_len, base, qual, base_len);

    return Status::OK();
  }

  Status FASTQIterator::read_line(const char **line_start, size_t *line_length, size_t skip_length)
  {
    // TODO efficiently check that the data_ has not been released
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

  void FASTQIterator::release() {
    if (fastq_file_) {
      fastq_file_->get()->release();
      fastq_file_->release();
    }
  }

} // namespace tensorflow {
