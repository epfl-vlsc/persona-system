#include <cstddef>
#include <string.h>
#include "buffer_list.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

  using namespace std;

  void BufferList::resize(size_t size) {
    if (size > buf_list_.size()) {
      // Can't do resize here because no default constructor
      for (decltype(size) i = buf_list_.size(); i < size; ++i) {
        buf_list_.push_back(unique_ptr<Buffer>(new Buffer()));
      }
    } else {
      buf_list_.resize(size);
    }
  }

  vector<unique_ptr<Buffer>>& BufferList::get() {
    return buf_list_;
  }

  Buffer* BufferList::get_at(size_t index) {
    if (index >= buf_list_.size())
      resize(index+1);
    return buf_list_[index].get();
  }

  void BufferList::reset() {
    for (auto &b : buf_list_) {
      b->reset();
    }
  }

} // namespace tensorflow {
