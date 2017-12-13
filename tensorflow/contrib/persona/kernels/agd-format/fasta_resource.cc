#include <utility>
#include "fasta_resource.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  FastaResource& FastaResource::operator=(FastaResource &&other) {
    fasta_file_ = move(other.fasta_file_);
    file_use_count_ = move(other.file_use_count_);
    done_ = move(other.done_);
    start_ptr_ = other.start_ptr_; other.start_ptr_ = nullptr;
    end_ptr_ = other.end_ptr_; other.end_ptr_ = nullptr;
    current_record_ = other.current_record_; other.current_record_ = nullptr;
    max_records_ = other.max_records_; other.max_records_ = 0;
    current_record_idx_ = other.current_record_idx_; other.current_record_idx_ = 0;
    return *this;
  }

  FastaResource::FastaResource(shared_ptr<FileResource> &fasta_file,
                               shared_ptr<atomic<unsigned int>> &use_count,
                               shared_ptr<volatile bool> &done,
                               const char *start_ptr, const char *end_ptr, const size_t max_records) :
    fasta_file_(fasta_file), file_use_count_(use_count), done_(done),
    start_ptr_(start_ptr), end_ptr_(end_ptr),
    current_record_(start_ptr),
    max_records_(max_records) {
    file_use_count_->fetch_add(1, memory_order::memory_order_release);
  }

  Status FastaResource::get_next_record(const char** bases, size_t* bases_len,
        const char** quals) {
    return Internal("unimplemented");
  }

  Status FastaResource::get_next_record(const char** bases, size_t* bases_len,
                                        const char** quals, const char** meta,
                                        size_t* meta_len) {

    if (!fasta_file_) {
      return Internal("get_next_record called with null data!");
    } else if (current_record_idx_ == max_records_) {
      return ResourceExhausted("no more records in this file");
    }
    *quals = nullptr;

    read_line(meta, meta_len, 1); // +1 to skip '>'
    read_lines(bases, bases_len); // bases or amino acids
    current_record_idx_++;

    return Status::OK();

  }

  Status FastaResource::get_next_record(Read &snap_read)
  {
      return Internal("get_next_record for snap read not implemented in FASTA resource");
  }

  void FastaResource::read_line(const char **line_start, size_t *line_length, size_t skip_length)
  {
    current_record_ += skip_length;
    *line_start = current_record_;
    size_t record_size = 0;

    for (; *current_record_ != '\n' && current_record_ < end_ptr_;
         record_size++, current_record_++);

    *line_length = record_size;
    current_record_++; // to skip over the '\n'
  }
  
  void FastaResource::read_lines(const char **line_start, size_t *line_length, size_t skip_length)
  {
    current_record_ += skip_length;
    *line_start = current_record_;
    size_t record_size = 0;

    // read to the next entry ('>') or end of file
    for (; *current_record_ != '>' && current_record_ < end_ptr_;
         record_size++, current_record_++);

    *line_length = record_size;
  }

  bool FastaResource::reset_iter()
  {
    current_record_ = start_ptr_;
    current_record_idx_ = 0;
    return true;
  }

  void FastaResource::release() {
    auto count = file_use_count_->fetch_sub(1, memory_order::memory_order_acquire);
    if (count == 1 && *done_) {
      fasta_file_->get()->release();
    }
    file_use_count_.reset();
    fasta_file_.reset();
    start_ptr_ = nullptr;
    end_ptr_ = nullptr;
    current_record_ = nullptr;
    max_records_ = 0;
    current_record_idx_ = 0;
  }

  size_t FastaResource::num_records() {
    return max_records_;
  }

} // namespace tensorflow {
