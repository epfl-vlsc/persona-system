#include <cstddef>
#include <string.h>
#include "buffer_list.h"
#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"

namespace tensorflow {

  using namespace std;

  void BufferList::resize(size_t size) {
    auto old_size = buf_list_.size();
    if (size > old_size) {
      buf_list_.resize(size);
    }
    reset_all();
    size_ = size;
  }

  size_t BufferList::size() const {
    return size_;
  }

  BufferPair& BufferList::operator[](size_t index) {
    if (index >= size_) {
      LOG(ERROR) << "FATAL: get_at requested index " << index << ", with only " << size_ << " elements. Real size: " << buf_list_.size();
    }
    // using at instead of operator[] because it will error here
    return buf_list_.at(index);
  }

  void BufferList::reset_all() {
    for (auto &b : buf_list_) {
      b.reset();
    }
  }

  void BufferList::reset() {
    reset_all();
    size_ = 0;
  }

} // namespace tensorflow {
