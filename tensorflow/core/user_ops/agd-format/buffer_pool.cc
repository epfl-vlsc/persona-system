#include "tensorflow/core/user_ops/object-pool/ref_pool_op.h"
#include "tensorflow/core/user_ops/agd-format/buffer.h"
#include "data.h"

namespace tensorflow {
  using namespace std;

  REGISTER_REFERENCE_POOL("BufferPool")
  .Doc(R"doc(
Creates and initializes a pool containing char buffers of size `buffer_size` bytes
  )doc");

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
