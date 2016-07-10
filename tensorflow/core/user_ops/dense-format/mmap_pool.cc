#include "tensorflow/core/user_ops/object-pool/ref_pool_op.h"
#include "tensorflow/core/user_ops/dna-align/data.h"
#include "shared_mmap_file_resource.h"

namespace tensorflow {
  using namespace std;

  REGISTER_REFERENCE_POOL("MMapPool")
  .Doc(R"doc(
Creates pools of MemoryMappedFile objects
)doc");


  class MMapPoolOp : public ReferencePoolOp<MemoryMappedFile, Data> {
  public:
    MMapPoolOp(OpKernelConstruction *ctx) : ReferencePoolOp<MemoryMappedFile, Data>(ctx) {}

  protected:
    unique_ptr<MemoryMappedFile> CreateObject() override {
      return unique_ptr<MemoryMappedFile>(new MemoryMappedFile());
    }
  };

  REGISTER_KERNEL_BUILDER(Name("MMapPool").Device(DEVICE_CPU), MMapPoolOp);
} // namespace tensorflow {
