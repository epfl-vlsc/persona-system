#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "dense-format/read_resource.h"
#include "object-pool/resource_container.h"
#include <cstdint>

namespace tensorflow {
  using namespace std;
  using namespace errors;

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
      const Tensor *input;
      OP_REQUIRES_OK(ctx, ctx->input("data", &input));
      auto data = input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &reads));
      core::ScopedUnref a(reads);
      ResourceReleaser<ReadResource> b(*reads);
      auto rs = reads->get();

      auto s = Status::OK();
      const char *base, *qual, *meta;
      size_t base_len, qual_len, meta_len;
      do {
        s = rs->get_next_record(&base, &base_len,
                                &qual, &qual_len,
                                &meta, &meta_len);
      } while (s.ok());

      if (!IsResourceExhausted(s)) {
        OP_REQUIRES_OK(ctx, s);
      }
    }
  };


REGISTER_KERNEL_BUILDER(Name("Sink").Device(DEVICE_CPU), SinkOp);
} // namespace tensorflow {
