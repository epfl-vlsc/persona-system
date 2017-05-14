#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"

namespace tensorflow {
  using namespace std;

  class BufferListPoolOp : public ReferencePoolOp<BufferList, BufferList> {
  public:
    BufferListPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<BufferList, BufferList>(ctx) {
    }

  protected:
    unique_ptr<BufferList> CreateObject() override {
      return unique_ptr<BufferList>(new BufferList());
    }
  private:
    TF_DISALLOW_COPY_AND_ASSIGN(BufferListPoolOp);
  };

  REGISTER_KERNEL_BUILDER(Name("BufferListPool").Device(DEVICE_CPU), BufferListPoolOp);
} // namespace tensorflow {
