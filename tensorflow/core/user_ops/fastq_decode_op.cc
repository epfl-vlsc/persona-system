
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

REGISTER_OP("DecodeFastq")
    .Input("read: string")
    .Output("output: string")
    .Doc(R"doc(
Convert fastq reads to tensors. A read batch maps to one tensor.

output: Each tensor will have the same shape as records.
)doc");

class DecodeFastqOp : public OpKernel {
 public:
  explicit DecodeFastqOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* read;

    OP_REQUIRES_OK(ctx, ctx->input("read", &read));

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, read->shape(), &output_tensor));

    // just copy over for now, should share same underlying storage as per tensor.h
    //LOG(INFO) << "read shape is " << read->shape().DebugString() << std::endl;

    OP_REQUIRES(ctx, output_tensor->CopyFrom(*read, read->shape()),
                  errors::InvalidArgument("DecodeFastq copy failed, input shape was ", 
                      read->shape().DebugString()));
  }
};

REGISTER_KERNEL_BUILDER(Name("DecodeFastq").Device(DEVICE_CPU), DecodeFastqOp);

}  // namespace tensorflow
