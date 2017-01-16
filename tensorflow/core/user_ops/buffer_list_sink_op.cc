#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "agd-format/buffer_list.h"
#include "object-pool/resource_container.h"
#include <cstdint>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  REGISTER_OP("BufferListSink")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("data: string")
  .Output("id: string")
  .Doc(R"doc(
Consumes the buffer input and produces nothing

Note that the output is meaningless. It's only purpose is so that
we can use it in other pipelines where writers are used
)doc");

  class BufferListSinkOp : public OpKernel {
  public:
    BufferListSinkOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      ResourceContainer<BufferList> *buf;
      const Tensor *input;
      OP_REQUIRES_OK(ctx, ctx->input("data", &input));
      auto data = input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &buf));
      buf->get()->wait_for_ready();
      core::ScopedUnref a(buf);
      {
        ResourceReleaser<BufferList> b(*buf); // make sure destructs first

        Tensor *output;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("id", TensorShape({}), &output));
        output->scalar<string>()() = "results sent to sink! you probably didn't want this";
      }
    }
  };


REGISTER_KERNEL_BUILDER(Name("BufferListSink").Device(DEVICE_CPU), BufferListSinkOp);
} // namespace tensorflow {
