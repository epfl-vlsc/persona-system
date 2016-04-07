#include "decompress.h"

#include <boost/iostreams/filtering_stream.hpp>

namespace tensorflow {

template <typename T>
Status decompressSegment(const char* segment,
                         const std::size_t segment_size,
                         std::vector<char> &output)
{
  namespace bsio = boost::iostreams;
  bsio::filtering_ostream os;
  os.push(T());
  os.push(std::back_inserter(output));
  os.write(segment, segment_size);

  // Not sure if these need to be called, or if the destructor gets them
  os.flush();
  os.reset();
  return Status::OK();
}

} // namespace tensorflow
