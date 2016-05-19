#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
  using namespace std;

  REGISTER_OP("SinkOp")
  .Attr("T: {float, int32, int64, string, float32}")
  .Input("data: T")
  .Doc(R"doc(
Consumes the input and produces nothing
)doc");

  class SinkOp : public OpKernel {
  public:
    SinkOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      // TODO do I have to even do anything with it?
      const Tensor *input_tensor;
      OP_REQUIRES_OK(ctx, ctx->input("data", &input_tensor));
    }
  };

#define REGISTER_TYPE(TYPE) \
  REGISTER_KERNEL_BUILDER(Name("SinkOp").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
                          SinkOp)

REGISTER_TYPE(string);
REGISTER_TYPE(int64);
REGISTER_TYPE(int32);
REGISTER_TYPE(float);
} // namespace tensorflow {
