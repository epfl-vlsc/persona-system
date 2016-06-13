/*
  An op to read from a queue of filename strings, and enqueue multiple shared resources corresponding to each file.
 */

#include "shared_mmap_file_resource.h"

namespace tensorflow {
  using namespace std;

  MemoryMappedFile::MemoryMappedFile(ResourceHandle &&file) : file_(move(file)) {}

  const char* MemoryMappedFile::data() const {
    return reinterpret_cast<const char*>(file_->data());
  }

  size_t MemoryMappedFile::size() const {
    return file_->length();
  }

  void MemoryMappedFile::own(ReadOnlyMemoryRegion *rmr) {
    file_.reset(rmr);
  }
} // namespace tensorflow {
