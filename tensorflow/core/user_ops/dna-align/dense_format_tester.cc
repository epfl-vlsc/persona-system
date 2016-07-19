#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/user_ops/dense-format/buffer.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include <cstdint>
#include "GenomeIndex.h"
#include "genome_index_resource.h"
#include "Read.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  REGISTER_OP("DenseTester")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("dense_buffer: string")
  .Input("sam_buffer: string")
  .Input("genome_handle: Ref(string)")
  .Doc(R"doc(
  Compares the dense format output with the SAM format output
)doc");

  class DenseTesterOp : public OpKernel {
  public:
    DenseTesterOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {

      // Get the dense format results in dense_buf
      // data(0) = container, data(1) = name
      ResourceContainer<Buffer> *dense_buf;
      const Tensor *dense_input;
      OP_REQUIRES_OK(ctx, ctx->input("dense_buffer", &dense_input));
      auto dense_data = dense_input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(dense_data(0), dense_data(1), &dense_buf));

      OP_REQUIRES_OK(ctx,
          GetResourceFromContext(ctx, "genome_handle", &genome_resource_));

      // TODO: is this the right way to get the sam_buffer?
      // Get the SAM format results in sam_buf
      ResourceContainer<Buffer> *sam_buf;
      const Tensor *sam_input;
      OP_REQUIRES_OK(ctx, ctx->input("sam_buffer", &sam_input));
      auto sam_data = sam_input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(sam_data(0), sam_data(1), &sam_buf));

    }

  private:      
    GenomeIndexResource* genome_resource_;
    Read sam_read_;
  };


REGISTER_KERNEL_BUILDER(Name("DenseTester").Device(DEVICE_CPU), DenseTesterOp);
} // namespace tensorflow
