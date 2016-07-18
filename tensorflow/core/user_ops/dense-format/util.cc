#include "util.h"
#include <cstring>

namespace tensorflow {
  using namespace std;

  Status copySegment(const char* segment,
                     const size_t segment_size,
                     vector<char> &output)
  {
    output.clear();
    output.reserve(segment_size);
    if (output.capacity() < segment_size) {
      // just use normal insert and not the optimized memcpy
      output.insert(output.end(), segment, segment+segment_size);
    } else {
      output.resize(segment_size);
      memcpy(&output[0], segment, segment_size);
    }
    return Status::OK();
  }

  Status appendSegment(const char* segment,
                       const size_t segment_size,
                       vector<char> &output, bool double_capacity)
  {
    size_t ideal_length = segment_size + output.size();
    if (double_capacity) {
      output.reserve(1); // just to make sure we don't multiply by 0!
      while (output.capacity() < ideal_length) {
        output.reserve(output.capacity() * 2);
      }
    } else {
      output.reserve(ideal_length);
    }
    if (output.capacity() < ideal_length) {
      output.insert(output.end(), segment, segment+segment_size);
    } else {
      auto old_size = output.size();
      output.resize(ideal_length);
      memcpy(&output[old_size], segment, segment_size);
    }
    return Status::OK();
  }
} // namespace tensorflow {
