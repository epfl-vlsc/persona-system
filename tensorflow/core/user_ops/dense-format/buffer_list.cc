#include <cstddef>
#include <string.h>
#include "buffer_list.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

  using namespace std;

  void BufferList::resize(size_t size) {
    buf_list_.resize(size);
  }

  std::vector<Buffer>& BufferList::get() {
    return buf_list_;
  }

  BufferList::~BufferList() {
    LOG(DEBUG) << "Calling ~BufferList(" << this << ")\n";
  }

  Buffer* BufferList::get_at(int index) {
    if (index >= buf_list_.size()) 
      buf_list_.resize(index+1);
    return &buf_list_[index];
  }

  void BufferList::reset() {
    for (auto &b : buf_list_) {
      b.reset();
    }
  }

} // namespace tensorflow {
