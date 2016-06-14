#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "dense_data.h"
#include <memory>

namespace tensorflow {
  REGISTER_OP("DenseRecordCreator")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("bases: string")
  .Input("qualities: string")
  .Output("dense_data: string")
  .Doc(R"doc(
Assembles the bases and qualities, based as opaque pointer scalar values,
into a DenseReadData object.

bases: the RecordParser object for the base records
qualities: the RecordParser object for the quality records
)doc");

  REGISTER_OP("DenseRecordAddMetadata")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("dense_data: string")
  .Input("metadata: string")
  .Output("dense_data_out: string")
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
#if 0
      const Tensor *base, *quality;
      OP_REQUIRES_OK(ctx, ctx->input("bases", &base));
      OP_REQUIRES_OK(ctx, ctx->input("qualities", &quality));

      const auto &base_shape = base->shape();
      // assume that the Python layer takes care of the shapes

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));
      auto rmgr = cinfo.resource_manager();

      auto base_matrix = base->matrix<string>();
      auto quality_matrix = quality->matrix<string>();
      string resource_name(name());

      Tensor *output;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("dense_data", base_shape, &output));

      auto output_matrix = output->matrix<string>();
      RecordParser *b, *q;
      DenseReadData *drd;
      for (int64 i = 0; i < base->dim_size(0); i++) {
        OP_REQUIRES_OK(ctx, rmgr->Lookup(base_matrix(i, 0), base_matrix(i, 1), &b));
        OP_REQUIRES_OK(ctx, rmgr->Lookup(quality_matrix(i, 0), quality_matrix(i, 1), &q));
        // No scoped unref. We need tihs to service inside of the DenseReadData object
        // We just need to get them out of this container

        resource_name = name();
        resource_name.append(to_string(round_++));

        drd = new DenseReadData(b, q);
        OP_REQUIRES_OK(ctx, rmgr->Create<DenseReadData>(cinfo.container(), resource_name, drd));

        output_matrix(i, 0) = cinfo.container();
        output_matrix(i, 1) = resource_name;

        OP_REQUIRES_OK(ctx, rmgr->Delete<RecordParser>(base_matrix(i, 0), base_matrix(i, 1)));
        OP_REQUIRES_OK(ctx, rmgr->Delete<RecordParser>(quality_matrix(i, 0), quality_matrix(i, 1)));
      }
#endif
    }
  private:
    size_t round_ = 0;
  };

  class DenseRecordAddMetadataOp : public OpKernel {
  public:
    DenseRecordAddMetadataOp(OpKernelConstruction *context) : OpKernel(context) {}
    void Compute(OpKernelContext* ctx) override {
#if 0
      const Tensor *dense_data, *metadata;
      OP_REQUIRES_OK(ctx, ctx->input("dense_data", &dense_data));
      OP_REQUIRES_OK(ctx, ctx->input("metadata", &metadata));
      // Assume that the Python ensures the proper shape

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));
      auto rmgr = cinfo.resource_manager();

      auto dense = dense_data->matrix<string>();
      auto md = metadata->matrix<string>();

      DenseReadData *drd;
      RecordParser *rp;
      for (size_t i = 0; i < dense.dimension(0); i++) {
        auto ctr = dense(i, 0);
        auto nm = dense(i, 1);
        OP_REQUIRES_OK(ctx, rmgr->Lookup(ctr, nm, &drd));
        OP_REQUIRES_OK(ctx, rmgr->Lookup(md(i, 0), md(i, 1), &rp));
        core::ScopedUnref unref_drd(drd);

        OP_REQUIRES_OK(ctx, drd->set_metadata(rp));

        // Just to delete from the resource manager. It's lifetime will be managed by drd
        OP_REQUIRES_OK(ctx, rmgr->Delete<RecordParser>(md(i, 0), md(i, 1)));
      }

      Tensor *dense_out;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("dense_data_out", dense_data->shape(), &dense_out));
      *dense_out = *dense_data;
#endif
    }
  };

  REGISTER_KERNEL_BUILDER(Name("DenseRecordCreator").Device(DEVICE_CPU), DenseRecordCreatorOp);
  REGISTER_KERNEL_BUILDER(Name("DenseRecordAddMetadata").Device(DEVICE_CPU), DenseRecordAddMetadataOp);
} // namespace tensorflow {
