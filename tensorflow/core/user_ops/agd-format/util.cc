#include "util.h"
#include <cstring>
#include <cstdint>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
  using namespace std;

  Status copySegment(const char* segment,
                     const size_t segment_size,
                     vector<char> &output)
  {
    output.clear();
    safe_reserve(output, segment_size);
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
    auto ideal_length = segment_size + output.size();
    auto init_capacity = output.capacity();

    safe_reserve(output, ideal_length);

    // double check if new capacity succeeds
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
