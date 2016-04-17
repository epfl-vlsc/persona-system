#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/user_ops/dna-align/snap_proto.pb.h"

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

    auto flat_bases = bases->flat<string>();
    auto flat_qualities = qualities->flat<string>();
    auto flat_metadata = metadata->flat<string>();


    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, bases->shape(), &output_tensor));
    auto out_flat = output_tensor->flat<string>();
    //OP_REQUIRES(ctx, output_tensor->CopyFrom(*read, read->shape()),
    //            errors::InvalidArgument("DecodeFastq copy failed, input shape was ", 
    //                                    read->shape().DebugString()));

    string s;
    for (size_t i = 0; i < flat_bases.size(); ++i) {
      SnapProto::AlignmentDef alignment;
      SnapProto::ReadDef* read = alignment.mutable_read();
      read->set_bases(flat_bases(i));
      read->set_meta(flat_metadata(i));
      read->set_length(flat_bases(i).length());
      read->set_qualities(flat_qualities(i));
      alignment.SerializeToString(&s);
      out_flat(i) = s;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseAggregator").Device(DEVICE_CPU), DenseAggregatorOp);

} // namespace tensorflow {
