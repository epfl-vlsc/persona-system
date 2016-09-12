#include "buffer.h"
#include <cstring>
#include <cstddef>
#include "data.h"
#include "util.h"
#include <chrono>

namespace tensorflow {

  using namespace std;
  using namespace errors;

  Buffer::Buffer(size_t initial_size, size_t extend_extra) : extend_extra_(extend_extra), size_(0), allocation_(initial_size) {
    // FIXME in an ideal world, allocation_ should be checked to be positive
    LOG(INFO) << "creating a new buffer!!";
    buf_.reset(new char[allocation_]());
  }

  Status Buffer::WriteBuffer(const char* content, size_t content_size) {
    if (allocation_ < content_size) {
      allocation_ = content_size + extend_extra_;
      LOG(INFO) << "REALLOCATING to WRITE  to " << allocation_ << " bytes";
      buf_.reset(new char[allocation_]()); // reset() -> old buf will be deleted
    }
    //auto t1 = std::chrono::high_resolution_clock::now();
    memcpy(buf_.get(), content, content_size);
    //auto t2 = std::chrono::high_resolution_clock::now();
    //auto writememcpytime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    //LOG(INFO) << "writebuf memcpy time: " << writememcpytime.count();
    size_ = content_size;
    return Status::OK();
  }

  Status Buffer::AppendBuffer(const char* content, size_t content_size) {
    auto old_size = size_;
    extend_size(content_size);
    memcpy(&buf_.get()[old_size], content, content_size);
    return Status::OK();
  }

  const char* Buffer::data() const {
    return buf_.get();
  }

  size_t Buffer::size() const {
    return size_;
  }

  void Buffer::reset() {
    size_ = 0;
  }

  char& Buffer::operator[](size_t idx) const {
    return buf_[idx];
  }

  void Buffer::reserve(size_t capacity) {
    if (capacity > allocation_) {
      allocation_ = capacity + extend_extra_;
      LOG(INFO) << "REALLOCATING WTF to " << allocation_ << " bytes";
      decltype(buf_) a(new char[allocation_]());
      memcpy(a.get(), buf_.get(), size_);
      buf_.swap(a);
    }
  }

  void Buffer::resize(size_t total_size) {
    reserve(total_size);
    size_ = total_size;
  }

  void Buffer::extend_allocation(size_t extend_size) {
    reserve(allocation_ + extend_size);
  }

  void Buffer::extend_size(size_t extend_size) {
    return resize(size_ + extend_size);
  }

} // namespace tensorflow {
