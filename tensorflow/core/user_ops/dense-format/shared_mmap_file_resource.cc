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

MappedFileRef::MappedFileRef(Tensor *ref_tensor) :
  ref_tensor_(ref_tensor) {}

const string& MappedFileRef::GetContainer()
{
  auto info = ref_tensor_->vec<string>();
  return info(kContainer);
}

void MappedFileRef::SetContainer(const string& container)
{
  auto info = ref_tensor_->vec<string>();
  info(kContainer) = container;
}

const string& MappedFileRef::GetName()
{
  auto info = ref_tensor_->vec<string>();
  return info(kName);
}

void MappedFileRef::SetName(const string& name)
{
  auto info = ref_tensor_->vec<string>();
  info(kName) = name;
}

ReadOnlyFileRef::ReadOnlyFileRef(const Tensor *ref_tensor) :
  ref_tensor_(ref_tensor) {}

const string& ReadOnlyFileRef::GetName()
{
  const auto info = ref_tensor_->vec<string>();
  return info(MappedFileRef::kName);
}

const string& ReadOnlyFileRef::GetContainer()
{
  const auto info = ref_tensor_->vec<string>();
  return info(MappedFileRef::kContainer);
}

} // namespace tensorflow {
