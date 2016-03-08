
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

REGISTER_OP("GenerateAlignCandidates")
    .Input("read: string")
    .Output("output: OUT_TYPE")
    .Attr("OUT_TYPE: list({string}) == 2")
    .Doc(R"doc(
Generates a set of alignment candidates for input `read`.

Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.

output: a list of tensors containing the read (0) and a [1] tensor 
containing the alignment candidates. 
)doc");

class GenerateAlignCandidatesOp : public OpKernel {
 public:
  explicit DecodeFastqOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));

    // other constructor stuff ...
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* read;

    // get the input tensor called "read"
    OP_REQUIRES_OK(ctx, ctx->input("read", &read));

    OpOutputList output;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));

    // allocate output tensor[0]
    // cannot allocate output tensor[1] yet because we 
    // don't know it's dims which will be [num_candidates]
    Tensor* out = nullptr;
    output.allocate(0, TensorShape(), &out);
    
    // copy over the read, should share same underlying storage as per tensor.h
    OP_REQUIRES(ctx, out->CopyFrom(*read, read->shape()),
                  errors::InvalidArgument("GenerateAlignCandidates copy failed, input shape was ", 
                      read->shape().DebugString()));
  }
 private:
  std::vector<DataType> out_type_;
};

REGISTER_KERNEL_BUILDER(Name("GenerateAlignCandidates").Device(DEVICE_CPU), GenerateAlignCandidatesOp);

}  // namespace tensorflow
