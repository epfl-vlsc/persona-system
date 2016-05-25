#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "dense_data.h"
#include <memory>

namespace tensorflow {
  REGISTER_OP("DenseRecordCreator")
  .Input("bases: int64")
  .Input("qualities: int64")
  .Output("dense_data: int64")
  .Doc(R"doc(
Assembles the bases and qualities, based as opaque pointer scalar values,
into a DenseReadData object.

bases: the RecordParser object for the base records
qualities: the RecordParser object for the quality records
)doc");

  REGISTER_OP("DenseRecordAddMetadata")
  .Input("dense_data: int64")
  .Input("metadata: int64")
  .Output("dense_data_out: int64")
  .Doc(R"doc(
dense_data: a DenseReadData object, as a pointer. Opaque to user-space (python)
metadata: a RecordParser object pointer for the metadata records
dense_data: the same data as dense_data input
)doc");

  using namespace std;
  using namespace errors;

  class DenseRecordCreatorOp : public OpKernel {
  public:
    DenseRecordCreatorOp(OpKernelConstruction *context) : OpKernel(context) {}
    void Compute(OpKernelContext* ctx) override {
      const Tensor *base, *quality;
      OP_REQUIRES_OK(ctx, ctx->input("bases", &base));
      OP_REQUIRES_OK(ctx, ctx->input("qualities", &quality));

      const auto &base_shape = base->shape();
      const auto &qual_shape = quality->shape();
      OP_REQUIRES(ctx, base_shape == qual_shape,
                  InvalidArgument("base shape (", base_shape.DebugString(), ") is not equal to qual shape (", qual_shape.DebugString(), ")"));

      Tensor *output;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("dense_data", base_shape, &output));

      auto bases = base->flat<int64>();
      auto qualities = quality->flat<int64>();
      auto out = output->flat<int64>();

      RecordParser *b, *q;
      DenseReadData *drd;
      for (size_t i = 0; i < bases.size(); i++) {
        b = reinterpret_cast<RecordParser*>(bases(i));
        q = reinterpret_cast<RecordParser*>(qualities(i));
        shared_ptr<RecordParser> b_sp(b), q_sp(q);
        drd = new DenseReadData(b_sp, q_sp);
        out(i) = reinterpret_cast<int64>(drd);
      }
    }
  };

  class DenseRecordAddMetadataOp : public OpKernel {
  public:
    DenseRecordAddMetadataOp(OpKernelConstruction *context) : OpKernel(context) {}
    void Compute(OpKernelContext* ctx) override {
      const Tensor *dense_data, *metadata;
      OP_REQUIRES_OK(ctx, ctx->input("dense_data", &dense_data));
      OP_REQUIRES_OK(ctx, ctx->input("metadata", &metadata));
      const auto &dense_shape = dense_data->shape();
      const auto &metadata_shape = metadata->shape();

      OP_REQUIRES(ctx, dense_shape == metadata_shape, InvalidArgument("Tensor shape for dense_data (", dense_shape.DebugString(), ") does not match metadata_shape (", metadata_shape.DebugString(), ")"));

      auto dense = dense_data->flat<int64>();
      auto md = metadata->flat<int64>();

      DenseReadData *read;
      RecordParser *rp;
      for (size_t i = 0; i < dense.size(); i++) {
        read = reinterpret_cast<DenseReadData *>(dense(i));
        rp = reinterpret_cast<RecordParser *>(md(i));
        shared_ptr<RecordParser> a(rp);
        OP_REQUIRES_OK(ctx, read->set_metadata(a));
      }

      Tensor *dd_out;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("dense_data_out", dense_shape, &dd_out));
      *dd_out = *dense_data;
    }
  };

  REGISTER_KERNEL_BUILDER(Name("DenseRecordCreator").Device(DEVICE_CPU), DenseRecordCreatorOp);
  REGISTER_KERNEL_BUILDER(Name("DenseRecordAddMetadata").Device(DEVICE_CPU), DenseRecordAddMetadataOp);
} // namespace tensorflow {
