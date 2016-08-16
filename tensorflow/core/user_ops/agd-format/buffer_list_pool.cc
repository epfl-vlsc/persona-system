#include "tensorflow/core/user_ops/object-pool/ref_pool_op.h"
#include "tensorflow/core/user_ops/agd-format/buffer_list.h"

namespace tensorflow {
  using namespace std;

  REGISTER_REFERENCE_POOL("BufferListPool")
  .Doc(R"doc(
Creates and initializes a pool containing a list of char buffers of size `buffer_size` bytes
  )doc");

  class BufferListPoolOp : public ReferencePoolOp<BufferList, BufferList> {
  public:
    BufferListPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<BufferList, BufferList>(ctx) {
    }

  protected:
    unique_ptr<BufferList> CreateObject() override {
      return unique_ptr<BufferList>(new BufferList());
    }
  };

  REGISTER_KERNEL_BUILDER(Name("BufferListPool").Device(DEVICE_CPU), BufferListPoolOp);
} // namespace tensorflow {
