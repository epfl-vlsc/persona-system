#include "fasta_chunker.h"

namespace tensorflow {
  using namespace std;

  // note: copies the shared ptr and any custom deleter (which we'll use)
  FastaChunker::FastaChunker(shared_ptr<FastaResource::FileResource> &data, const size_t chunk_size) :
    data_(data), chunk_size_(chunk_size) {
    auto *file_data = data->get();
    current_ptr_ = file_data->data();
    end_ptr_ = current_ptr_ + file_data->size();
    file_use_count_ = make_shared<atomic<unsigned int>>(0);
    done_flag_ = make_shared<volatile bool>(false);
    *done_flag_ = false;
  }

  bool FastaChunker::next_chunk(FastaResource &resource) {
    const char *record_base = current_ptr_;
    size_t record_count = 0;
    while (record_count < chunk_size_ && advance_record()) {
      record_count++;
    }

    // happens if the underlying pointer arithmetic detectse that this is already exhausted
    if (record_count == 0) {
      *done_flag_ = true;
      return false;
    }

    //create a fasta resource
    resource = FastaResource(data_, file_use_count_, done_flag_, record_base, current_ptr_, record_count);

    return true;
  }

  // just assume the basic 4-line format for now
  bool FastaChunker::advance_record() {
    // advance 1 line for the `>' metadata/name, which we assume is only one line
    if (!advance_line()) {
      return false;
    }

    // sequence data can be multiple lines (WHYYYYYYY)
    while (*current_ptr_ != '>' && current_ptr_ != end_ptr_) {
      if (!advance_line()) {
        return false;
      }
    }
    return true;
  }

  bool FastaChunker::advance_line() {
    if (current_ptr_ == end_ptr_) {
      return false;
    } else {
      while (current_ptr_ < end_ptr_ && *current_ptr_ != '\n') {
        current_ptr_++; // yes, you can put this in the 2nd clause expression, but that is confusing
      }

      // in this case, we want to advance OVER the '\n', as that is what caused us to exit the loop
      if (current_ptr_ < end_ptr_) {
        current_ptr_++;
      }
      return true;
    }
  }

} // namespace tensorflow {
