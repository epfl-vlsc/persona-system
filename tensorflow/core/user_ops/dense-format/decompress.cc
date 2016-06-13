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

Status copySegment(const char* segment,
                   const size_t segment_size,
                   vector<char> &output)
{
  output.reserve(segment_size);
  if (output.capacity() < segment_size) {
    // just use normal insert and not the optimized memcpy
    output.insert(output.end(), segment, segment+segment_size);
  } else {
    memcpy(&output[0], segment, segment_size);
    output.resize(segment_size);
  }
  return Status::OK();
}

} // namespace tensorflow
