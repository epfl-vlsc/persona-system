#include "read_resource.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  ReadResource::~ReadResource() {}

  Status ReadResource::split(size_t chunk, BufferList *bl)
  {
    return Unimplemented("resource splitting not supported for this resource");
  }

  Status ReadResource::get_next_subchunk(ReadResource **rr, BufferPair **b)
  {
    return Unimplemented("resource splitting not supported for this resource");
  }

  bool ReadResource::reset_iter() {
    return false;
  }

  size_t ReadResource::num_records() {
    return 0;
  }

  void ReadResource::release() {}

  ReadResourceReleaser::ReadResourceReleaser(ReadResource &r) : rr_(r) {}

  ReadResourceReleaser::~ReadResourceReleaser()
  {
    rr_.release();
  }
} // namespace tensorflow {
