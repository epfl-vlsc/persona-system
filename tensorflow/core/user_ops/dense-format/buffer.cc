#include <cstddef>
#include <string.h>
#include "tensorflow/core/user_ops/dna-align/data.h"
#include "tensorflow/core/user_ops/dense-format/buffer.h"
#include "util.h"

namespace tensorflow {

  using namespace std;

  Status Buffer::WriteBuffer(const char* content, size_t content_size) {
    return copySegment(content, content_size, buf_);
  }

  Status Buffer::AppendBuffer(const char* content, size_t content_size) {
    return appendSegment(content, content_size, buf_);
  }

  const char* Buffer::data() const {
    return &buf_[0];
  }

  size_t Buffer::size() const {
    return buf_.size();
  }

} // namespace tensorflow {
