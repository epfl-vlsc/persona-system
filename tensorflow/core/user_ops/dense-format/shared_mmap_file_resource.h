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
#include "data.h"

namespace tensorflow {
  /*
    Just a convenience class to track a read-only memory region throughout the execution context.
   */

  class MemoryMappedFile : public Data {
  public:
    typedef std::unique_ptr<ReadOnlyMemoryRegion> ResourceHandle;

    virtual const char* data() const override;
    virtual std::size_t size() const override;

    // needed for pool creation
    MemoryMappedFile() = default;
    MemoryMappedFile(ResourceHandle &&file);
    MemoryMappedFile& operator=(MemoryMappedFile &&x) = default;

    void own(ReadOnlyMemoryRegion *rmr);
  private:
    ResourceHandle file_;
  };

} // namespace tensorflow {

#endif
