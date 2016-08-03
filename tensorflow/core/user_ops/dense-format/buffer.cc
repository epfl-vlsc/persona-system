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

  Status Buffer::AppendBufferDouble(const char* content, size_t content_size) {
    return appendSegment(content, content_size, buf_, true);
  }

  const char* Buffer::data() const {
    return &buf_[0];
  }

  size_t Buffer::size() const {
    return buf_.size();
  }

  void Buffer::reset() {
    buf_.clear();
    data_ready_ = false;
  }

  void Buffer::set_ready() {
    data_ready_ = true;
    ready_cv_.notify_all();
  }

  decltype(Buffer::buf_)& Buffer::get() {
    return buf_;
  }

  decltype(Buffer::buf_)& Buffer::get_when_ready() {
    if (!data_ready_) {
      mutex_lock l(mu_);
      ready_cv_.wait(l, [this]() {
          return data_ready_;
        });
    }
    return buf_;
  }
} // namespace tensorflow {
