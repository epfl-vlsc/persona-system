#include "data.h"
#include "fasta_resource.h"

namespace tensorflow {

  class FastaChunker {
  public:

    FastaChunker(std::shared_ptr<FastaResource::FileResource> &data, const std::size_t chunk_size);

    // Assigns a fastq resource to the parameter
    // bool indicates whether this operation was successful
    // false = no more chunks
    bool next_chunk(FastaResource &resource);

  private:

    bool create_chunk(FastaResource &resource);
    bool advance_record();
    bool advance_line();

    const char* skip_line();

    std::shared_ptr<FastaResource::FileResource> data_;
    std::shared_ptr<std::atomic<unsigned int>> file_use_count_;
    std::shared_ptr<volatile bool> done_flag_;
    const char *current_ptr_, *end_ptr_;
    std::size_t chunk_size_;
  };

} // namespace tensorflow {
