#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "dense-format/read_resource.h"
#include "object-pool/resource_container.h"

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
      ResourceContainer<ReadResource> *reads;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "data", &reads));
      core::ScopedUnref a(reads);
      ResourceReleaser<ReadResource> b(*reads);
      reads->get()->release();
    }
  };


REGISTER_KERNEL_BUILDER(Name("Sink").Device(DEVICE_CPU), SinkOp);
} // namespace tensorflow {
