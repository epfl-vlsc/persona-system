#include "read_resource.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  ReadResource::~ReadResource() {}

  Status ReadResource::get_next_record(const char **bases, std::size_t *bases_length,
                                const char **qualities, std::size_t *qualities_length)
  {
    return Unimplemented("partial record reading not suppported");
  }

  Status ReadResource::split(size_t chunk, vector<unique_ptr<ReadResource>> &split_resources)
  {
    return Unimplemented("resource splitting not supported for this resource");
  }

  bool ReadResource::has_qualities() {
    return false;
  }

  bool ReadResource::has_metadata() {
    return false;
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
