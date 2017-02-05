#include "data.h"
#include "fastq_resource.h"

namespace tensorflow {

  class FastqChunker {
  public:

    FastqChunker(std::shared_ptr<FastqResource::FileResource> &data, const std::size_t chunk_size);

    // Assigns a fastq resource to the parameter
    // bool indicates whether this operation was successful
    // false = no more chunks
    bool next_chunk(FastqResource &resource);

  private:

    bool create_chunk(FastqResource &resource);
    bool advance_record();
    bool advance_line();

    const char* skip_line();

    std::shared_ptr<FastqResource::FileResource> data_;
    std::shared_ptr<std::atomic<unsigned int>> file_use_count_;
    std::shared_ptr<volatile bool> done_flag_;
    const char *current_ptr_, *end_ptr_;
    std::size_t chunk_size_;
  };

} // namespace tensorflow {