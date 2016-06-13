#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "dense-format/dense_data.h"
#include "dense-format/parser.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  REGISTER_OP("Sink")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("data: string")
  .Doc(R"doc(
Consumes the input and produces nothing
)doc");

  class SinkOp : public OpKernel {
  public:
    SinkOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      // TODO do I have to even do anything with it?
      const Tensor *input_tensor;
      OP_REQUIRES_OK(ctx, ctx->input("data", &input_tensor));
      // assume that the Python layer checks the shape, se we don't have to do it here

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));
      auto rmgr = cinfo.resource_manager();

      ResourceContainer<RecordParser> *rp;
      auto input = input_tensor->matrix<string>();
      for (int64 i = 0; i < input_tensor->dim_size(0); i++) {
        auto ctr = input(i, 0);
        auto nm = input(i, 1);
        OP_REQUIRES_OK(ctx, rmgr->Lookup<ResourceContainer<RecordParser>>(ctr, nm, &rp));
        // TODO need to unref rp here!
        rp->release();
      }
    }
  };


REGISTER_KERNEL_BUILDER(Name("Sink").Device(DEVICE_CPU), SinkOp);
} // namespace tensorflow {
