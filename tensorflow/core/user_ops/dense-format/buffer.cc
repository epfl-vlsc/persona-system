#include <cstddef>
#include <string.h>
#include "data.h"
#include "buffer.h"
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

  void Buffer::reset() {
    buf_.clear();
  }

  vector<char>& Buffer::get() {
    return buf_;
  }

} // namespace tensorflow {
