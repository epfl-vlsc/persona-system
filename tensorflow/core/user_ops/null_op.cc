#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/user_ops/dense-format/parser.h"

namespace tensorflow {
  using namespace std;

  REGISTER_OP("Sink")
  .Attr("T: {float, int32, int64, string, float32}")
  .Input("data: T")
  .Doc(R"doc(
Consumes the input and produces nothing
)doc");

  REGISTER_OP("DeleteColumn")
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

  class DeleteColumnOp : public OpKernel {
  public:
    DeleteColumnOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      const Tensor *input_tensor;
      OP_REQUIRES_OK(ctx, ctx->input("data", &input_tensor));
      auto xs = input_tensor->flat<int64>();
      RecordParser *rp;
      for (int i = 0; i < xs.size(); i++) {
        rp = reinterpret_cast<RecordParser*>(xs(i));
        delete rp;
      }
        //RecordParser *rp;
      /*
      auto flat = input_tensor->vec<int64>();
      for (int i = 0; i < 3; i++) {
        auto x = reinterpret_cast<RecordParser*>(flat(i));
        delete x;
      }
      */
    }
  };

REGISTER_KERNEL_BUILDER(Name("DeleteColumn").Device(DEVICE_CPU), DeleteColumnOp);

#define REGISTER_TYPE(TYPE) \
  REGISTER_KERNEL_BUILDER(Name("Sink").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
                          SinkOp)

REGISTER_TYPE(string);
REGISTER_TYPE(int64);
REGISTER_TYPE(int32);
REGISTER_TYPE(float);
} // namespace tensorflow {
