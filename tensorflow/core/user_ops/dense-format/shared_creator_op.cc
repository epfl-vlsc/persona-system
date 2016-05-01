/*
  An op to read from a queue of filename strings, and enqueue multiple shared resources corresponding to each file.
 */

#include <memory>
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
  using namespace std;

  /*
    Just a convenience class to track a read-only memory region throughout the execution context.
   */
  class MemoryMappedFile : public ResourceBase {
  public:
    typedef shared_ptr<ReadOnlyMemoryRegion> ResourceHandle;

    explicit MemoryMappedFile(ResourceHandle &mapped_file) :
      file_(mapped_file) {}

    ReadOnlyMemoryRegion *
    GetMappedRegion() {
      return file_.get();
    }

    string DebugString() override
    {
      return "a Memory Mapped File";
    }

  private:
       ResourceHandle file_;
  };
} // namespace tensorflow {
