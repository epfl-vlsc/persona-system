#include <cstddef>
#include <string.h>
#include "buffer_list.h"

namespace tensorflow {

  using namespace std;

  void BufferList::resize(size_t size) {
    buf_list_.resize(size);
  }

  std::vector<Buffer>& BufferList::get() {
    return buf_list_;
  }

  Buffer& BufferList::get_at(int index) {
    if (index >= buf_list_.size()) 
      buf_list_.resize(index+1);
    return buf_list_[index];
  }

  void BufferList::reset() {
    buf_list_.clear();
  }

} // namespace tensorflow {
