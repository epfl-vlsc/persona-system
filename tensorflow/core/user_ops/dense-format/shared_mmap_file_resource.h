/*
  An op to read from a queue of filename strings, and enqueue multiple shared resources corresponding to each file.
 */

#ifndef TENSORFLOW_CORE_USEROPS_DENSE_ALIGN_MMAP_H_
#define TENSORFLOW_CORE_USEROPS_DENSE_ALIGN_MMAP_H_

#include <memory>
#include <cstdint>
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
  /*
    Just a convenience class to track a read-only memory region throughout the execution context.
   */
  class MemoryMappedFile : public ResourceBase {
  public:
    typedef std::shared_ptr<ReadOnlyMemoryRegion> ResourceHandle;

    explicit MemoryMappedFile(ResourceHandle &mapped_file);

    ReadOnlyMemoryRegion* GetMappedRegion();

    string DebugString() override;

    ReadOnlyMemoryRegion* get();
  private:
       ResourceHandle file_;
  };
} // namespace tensorflow {

#endif
