#include "decompress.h"

#include <cstring>
#include <boost/iostreams/filtering_stream.hpp>

namespace tensorflow {

  //auto decompressBZIP2 = decompressSegment<boost::iostreams::bzip2_decompressor>;
  using namespace std;

template <typename T>
Status decompressSegment(const char* segment,
                         const size_t segment_size,
                         vector<char> &output)
{
  namespace bsio = boost::iostreams;
  bsio::filtering_ostream os;

  // No need to call output.reserve().
  // We can't be sure exactly how much data it will produce.

  os.push(T());
  os.push(back_inserter(output));
  os.write(segment, segment_size);

  // Not sure if these need to be called, or if the destructor gets them
  os.flush();
  os.reset();
  return Status::OK();
}

template
Status decompressSegment<boost::iostreams::bzip2_decompressor>(const char* segment,
                                                                const size_t segment_size,
                                                                vector<char> &output);

template
Status decompressSegment<boost::iostreams::gzip_decompressor>(const char* segment,
                                                                const size_t segment_size,
                                                                vector<char> &output);
} // namespace tensorflow
