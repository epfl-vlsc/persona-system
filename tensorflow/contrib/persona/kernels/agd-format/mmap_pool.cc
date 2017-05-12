#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "data.h"
#include "shared_mmap_file_resource.h"

namespace tensorflow {
  using namespace std;


  class MMapPoolOp : public ReferencePoolOp<MemoryMappedFile, Data> {
  public:
    MMapPoolOp(OpKernelConstruction *ctx) : ReferencePoolOp<MemoryMappedFile, Data>(ctx) {}

  protected:
    unique_ptr<MemoryMappedFile> CreateObject() override {
      return unique_ptr<MemoryMappedFile>(new MemoryMappedFile());
    }

  private:
    TF_DISALLOW_COPY_AND_ASSIGN(MMapPoolOp);
  };

  REGISTER_KERNEL_BUILDER(Name("MMapPool").Device(DEVICE_CPU), MMapPoolOp);
} // namespace tensorflow {
