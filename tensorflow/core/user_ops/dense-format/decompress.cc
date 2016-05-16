#include "decompress.h"

#include <boost/iostreams/filtering_stream.hpp>

namespace tensorflow {

  //auto decompressBZIP2 = decompressSegment<boost::iostreams::bzip2_decompressor>;

template <typename T>
Status decompressSegment(const char* segment,
                         const std::size_t segment_size,
                         std::vector<char> &output)
{
  namespace bsio = boost::iostreams;
  bsio::filtering_ostream os;

  // No need to call output.reserve().
  // We can't be sure exactly how much data it will produce.

  os.push(T());
  os.push(std::back_inserter(output));
  os.write(segment, segment_size);

  // Not sure if these need to be called, or if the destructor gets them
  os.flush();
  os.reset();
  return Status::OK();
}

template
Status decompressSegment<boost::iostreams::bzip2_decompressor>(const char* segment,
                                                                const std::size_t segment_size,
                                                                std::vector<char> &output);

template
Status decompressSegment<boost::iostreams::gzip_decompressor>(const char* segment,
                                                                const std::size_t segment_size,
                                                                std::vector<char> &output);

Status copySegment(const char* segment,
                   const std::size_t segment_size,
                   std::vector<char> &output)
{
  output.reserve(segment_size);
  output.insert(output.end(), segment, segment+segment_size);
  return Status::OK();
}

} // namespace tensorflow
