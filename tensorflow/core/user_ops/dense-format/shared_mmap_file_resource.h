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

  class MappedFileRef {
  public:
    MappedFileRef(Tensor *ref_tensor);

    const string& GetContainer();
    void SetContainer(const string& container);
    const string& GetName();
    void SetName(const string& name);

  private:
    Tensor *ref_tensor_;

  public:
    static const std::size_t kContainer = 0;
    static const std::size_t kName = 1;
  };

  class ReadOnlyFileRef {
  public:
    ReadOnlyFileRef(const Tensor *ref_tensor);
    const string& GetName();
    const string& GetContainer();
  private:
    const Tensor *ref_tensor_;
  };

} // namespace tensorflow {

#endif
