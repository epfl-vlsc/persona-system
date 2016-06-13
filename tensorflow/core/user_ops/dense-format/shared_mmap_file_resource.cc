/*
  An op to read from a queue of filename strings, and enqueue multiple shared resources corresponding to each file.
 */

#include "shared_mmap_file_resource.h"

namespace tensorflow {

MemoryMappedFile::MemoryMappedFile(ResourceHandle &mapped_file) :
  file_(mapped_file) {}

ReadOnlyMemoryRegion *
MemoryMappedFile::GetMappedRegion() {
  return file_.get();
}

ReadOnlyMemoryRegion* MemoryMappedFile::get()
{
  return file_.get();
}

string MemoryMappedFile::DebugString()
{
  return "a Memory Mapped File";
}

} // namespace tensorflow {
