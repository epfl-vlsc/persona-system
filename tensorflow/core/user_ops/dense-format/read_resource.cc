#include "read_resource.h"

namespace tensorflow {
  ReadResource::~ReadResource() {}

  bool ReadResource::has_qualities() {
    return false;
  }

  bool ReadResource::has_metadata() {
    return false;
  }

  bool ReadResource::reset_iter() {
    return false;
  }
} // namespace tensorflow {
