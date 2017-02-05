#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "buffer.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include <cstdint>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class BufferSinkOp : public OpKernel {
  public:
    BufferSinkOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      ResourceContainer<Data> *buf;
      const Tensor *input;
      OP_REQUIRES_OK(ctx, ctx->input("data", &input));
      auto data = input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &buf));
      core::ScopedUnref a(buf);
      {
        ResourceReleaser<Data> b(*buf); // make sure destructs first
      }
    }
  };


REGISTER_KERNEL_BUILDER(Name("BufferSink").Device(DEVICE_CPU), BufferSinkOp);
} // namespace tensorflow {
