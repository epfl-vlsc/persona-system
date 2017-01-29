#include <utility>
#include "fastq_resource.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  FastqResource& FastqResource::operator=(FastqResource &&other) {
    fastq_file_ = move(other.fastq_file_);
    start_ptr_ = other.start_ptr_; other.start_ptr_ = nullptr;
    end_ptr_ = other.end_ptr_; other.end_ptr_ = nullptr;
    current_record_ = other.current_record_; other.current_record_ = nullptr;
    max_records_ = other.max_records_; other.max_records_ = 0;
    current_record_idx_ = other.current_record_idx_; other.current_record_idx_ = 0;
    return *this;
  }

  FastqResource::FastqResource(std::shared_ptr<FileResource> &fastq_file, const char *start_ptr,
                               const char *end_ptr, const std::size_t max_records) : fastq_file_(fastq_file),
                                                                                     start_ptr_(start_ptr), end_ptr_(end_ptr),
                                                                                     current_record_(start_ptr),
                                                                                     max_records_(max_records) {}

  Status FastqResource::get_next_record(const char** bases, size_t* bases_len,
        const char** quals) {
    return Internal("unimplemented");
  }

  Status FastqResource::get_next_record(Read &snap_read)
  {
    if (!fastq_file_) {
      return Internal("get_next_record called with null data!");
    } else if (current_record_idx_ == max_records_) {
      return ResourceExhausted("no more records in this file");
    }

    const char *meta, *base, *qual;
    size_t meta_len, base_len;

    read_line(&meta, &meta_len, 1); // +1 to skip '@'
    read_line(&base, &base_len);
    skip_line();
    read_line(&qual, &base_len);
    snap_read.init(meta, meta_len, base, qual, base_len);
    current_record_idx_++;

    return Status::OK();
  }

  void FastqResource::read_line(const char **line_start, size_t *line_length, size_t skip_length)
  {
    current_record_ += skip_length;
    *line_start = current_record_;
    size_t record_size = 0;

    for (; *current_record_ != '\n' && current_record_ < end_ptr_;
         record_size++, current_record_++);

    *line_length = record_size;
    current_record_++; // to skip over the '\n'
  }

  void FastqResource::skip_line()
  {
    for (; *current_record_ != '\n' && current_record_ < end_ptr_; current_record_++);
    /* include this check if we want to avoid leaving the pointers in an invalid state
       currently they won't point to anything, so it should be fine
    if (current_record_ < end_ptr_) {
      current_record_++;
    }
    */
    current_record_++; // to skip over the '\n'
  }

  bool FastqResource::reset_iter()
  {
    current_record_ = start_ptr_;
    current_record_idx_ = 0;
    return true;
  }

  void FastqResource::release() {
    fastq_file_.reset();
  }

} // namespace tensorflow {
