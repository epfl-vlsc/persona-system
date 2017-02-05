#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "data.h"

namespace tensorflow {
  using namespace std;

  class BufferPoolOp : public ReferencePoolOp<Buffer, Data> {
  public:
    BufferPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<Buffer, Data>(ctx) {
    }

  protected:
    unique_ptr<Buffer> CreateObject() override {
      return unique_ptr<Buffer>(new Buffer());
    }
  };

  REGISTER_KERNEL_BUILDER(Name("BufferPool").Device(DEVICE_CPU), BufferPoolOp);
} // namespace tensorflow {
