#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/user_ops/dna-align/snap_read_decode.h"

namespace tensorflow {

  REGISTER_OP("DenseAggregator")
    .Input("bases: string")
    .Input("qualities: string")
    .Input("metadata: string")
    .Output("read_record: string")
    .Doc(R"doc(
      An op that aggregates three streams from the DenseFile format and outputs
      a SnapProto-serialized string, for use in the Aligner.

      bases: a string generated from a BaseReader
      qualities: a string generated from a DenseReader
      metadata: a string generated from a DenseReader
    )doc");

class DenseAggregatorOp : public OpKernel {
public:
  explicit DenseAggregatorOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    using namespace std;

    const Tensor *bases, *qualities, *metadata;
    OP_REQUIRES_OK(ctx, ctx->input("bases", &bases));
    OP_REQUIRES_OK(ctx, ctx->input("qualities", &qualities));
    OP_REQUIRES_OK(ctx, ctx->input("metadata", &metadata));

    // Now verify that they are all the same dimension
    // Might be able to not worry about this for now, but let's just assume this
    OP_REQUIRES(ctx, bases->IsSameSize(*qualities) && bases->IsSameSize(*metadata),
                errors::InvalidArgument("Unequal DenseAggregator Shapes\nBases: ",
                                        bases->DebugString(), "\nQualities: ",
                                        qualities->DebugString(), "\nMetadata: ",
                                        metadata->DebugString()));

    auto flat_bases = bases->vec<string>();
    auto flat_qualities = qualities->vec<string>();
    auto flat_metadata = metadata->vec<string>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({3, 2}), &output_tensor));

    auto output = output_tensor->matrix<string>();
    output(0, 0) = flat_bases(0);
    output(0, 1) = flat_bases(1);
    output(1, 0) = flat_qualities(0);
    output(1, 1) = flat_qualities(1);
    output(2, 0) = flat_metadata(0);
    output(2, 1) = flat_metadata(1);
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseAggregator").Device(DEVICE_CPU), DenseAggregatorOp);

} // namespace tensorflow {
