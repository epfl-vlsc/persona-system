#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/user_ops/dense-format/parser.h"

namespace tensorflow {
  using namespace std;

  REGISTER_OP("SinkOp")
  .Attr("T: {float, int32, int64, string, float32}")
  .Input("data: T")
  .Doc(R"doc(
Consumes the input and produces nothing
)doc");

  REGISTER_OP("DeleteOp")
  .Input("data: int64")
  .Doc(R"doc(
Deletes the triple produced by the concat op for the dense op

This is really hacky, and is basically just for a proof-of-concept
of the NULL pipeline.
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

  class DeleteOp : public OpKernel {
  public:
    DeleteOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      const Tensor *input_tensor;
      OP_REQUIRES_OK(ctx, ctx->input("data", &input_tensor));
      OP_REQUIRES(ctx, input_tensor->shape() == TensorShape({3}),
                  Internal("TensorShape of DeleteOp is wrong")
                  );
      auto x = input_tensor->scalar<int64>();
      auto y = reinterpret_cast<RecordParser*>(x());
      delete y;
      /*
      auto flat = input_tensor->vec<int64>();
      for (int i = 0; i < 3; i++) {
        auto x = reinterpret_cast<RecordParser*>(flat(i));
        delete x;
      }
      */
    }
  };

REGISTER_KERNEL_BUILDER(Name("DeleteOp").Device(DEVICE_CPU), DeleteOp);

#define REGISTER_TYPE(TYPE) \
  REGISTER_KERNEL_BUILDER(Name("SinkOp").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
                          SinkOp)

REGISTER_TYPE(string);
REGISTER_TYPE(int64);
REGISTER_TYPE(int32);
REGISTER_TYPE(float);
} // namespace tensorflow {
