
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include <iostream>
namespace tensorflow {

REGISTER_OP("DecodeFastq")
    .Input("reads: string")
    .Output("output: string")
    .Doc(R"doc(
Convert fastq reads to tensors. A read batch maps to one tensor.

output: Each tensor will have the same shape as records.
)doc");

class DecodeFastqOp : public OpKernel {
 public:
  explicit DecodeFastqOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor* reads;

    OP_REQUIRES_OK(ctx, ctx->input("reads", &reads));

    auto reads_t = reads->flat<string>();
    int reads_size = reads_t.size();
    LOG(INFO) << "reads_size is " << reads_size;

    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape(), &output_tensor));
    auto output = output_tensor->template scalar<string>();

    output() = "Just testing decode fastq ...";

    for (int i = 0; i < reads_size; i++) {
        const StringPiece read(reads_t(i));
        std::cout << "printing read: " << read << std::endl;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DecodeFastq").Device(DEVICE_CPU), DecodeFastqOp);

}  // namespace tensorflow
