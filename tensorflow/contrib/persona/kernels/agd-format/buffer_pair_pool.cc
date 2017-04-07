#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"

namespace tensorflow {
  using namespace std;

  class BufferPairPoolOp : public ReferencePoolOp<BufferPair, BufferPair> {
  public:
    BufferPairPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<BufferPair, BufferPair>(ctx) {
    }

  protected:
    unique_ptr<BufferPair> CreateObject() override {
      return unique_ptr<BufferPair>(new BufferPair());
    }
  };

  REGISTER_KERNEL_BUILDER(Name("BufferPairPool").Device(DEVICE_CPU), BufferPairPoolOp);
} // namespace tensorflow {
