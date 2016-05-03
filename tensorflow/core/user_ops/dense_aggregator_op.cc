#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/user_ops/dna-align/snap_read_decode.h"

namespace tensorflow {

  REGISTER_OP("DenseAggregator")
    .Input("bases: string")
    .Input("bases_count: int32")
    .Input("qualities: string")
    .Input("qualities_count: int32")
    .Input("metadata: string")
    .Input("metadata_count: int32")
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
    using namespace errors;

    const Tensor *bases, *bases_size, *qualities, *qualities_size, *metadata, *metadata_size;
    OP_REQUIRES_OK(ctx, ctx->input("bases", &bases));
    OP_REQUIRES_OK(ctx, ctx->input("qualities", &qualities));
    OP_REQUIRES_OK(ctx, ctx->input("metadata", &metadata));
    OP_REQUIRES_OK(ctx, ctx->input("bases_count", &bases_size));
    OP_REQUIRES_OK(ctx, ctx->input("qualities_count", &qualities_size));
    OP_REQUIRES_OK(ctx, ctx->input("metadata_count", &metadata_size));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bases->shape()) && TensorShapeUtils::IsVector(qualities->shape()) && TensorShapeUtils::IsVector(metadata->shape()),
                InvalidArgument("Not all inputs are vectors"));

    // Now verify that they are all the same dimension
    // Might be able to not worry about this for now, but let's just assume this
    OP_REQUIRES(ctx, bases->IsSameSize(*qualities) && bases->IsSameSize(*metadata),
                InvalidArgument("Unequal DenseAggregator Shapes\nBases: ",
                                        bases->DebugString(), "\nQualities: ",
                                        qualities->DebugString(), "\nMetadata: ",
                                        metadata->DebugString()));

    // TODO enforce shape on the count things?
    auto bases_count = bases_size->scalar<int32>()();
    auto qualities_count = qualities_size->scalar<int32>()();
    auto metadata_count = metadata_size->scalar<int32>()();

    OP_REQUIRES(ctx, bases_count == qualities_count && bases_count == metadata_count,
                Internal("Differing counts for actual records:\nBases: ", bases_count, ", Qualities: ", qualities_count, ", Metadata: ", metadata_count));

    Tensor *output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("read_record", TensorShape({3, bases->dim_size(0)}), &output));
    MutableSnapReadDecode output_matrix(output);
    OP_REQUIRES(ctx, output_matrix.set_all_bases(*bases),
               Internal("Unable to set the bases in DenseAggregator"));
    OP_REQUIRES(ctx, output_matrix.set_all_qualities(*qualities),
               Internal("Unable to set the qualities in DenseAggregator"));
    OP_REQUIRES(ctx, output_matrix.set_all_metadata(*metadata),
               Internal("Unable to set the metadata in DenseAggregator"));
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseAggregator").Device(DEVICE_CPU), DenseAggregatorOp);

} // namespace tensorflow {
