#include "compress.h"

namespace tensorflow {

using namespace std;

Status compressGZIP(const char* segment,
                    const size_t segment_size,
                    vector<char> &output)
{
  return Status::OK(); // for now, just to compile
}

} // namespace tensorflow {
