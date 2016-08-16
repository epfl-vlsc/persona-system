#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/logging.h"
#include "buffer.h"
#include "compression.h"
#include "data.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include <string>

namespace tensorflow {

  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("GZIPDecomp");
  }

  REGISTER_OP(op_name.c_str())
  .Input("buffer_pool: Ref(string)")
  .Input("data_handle: string")
  .Output("uncomp_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Decompress the entire data buffer that this receives
)doc");


  class GZIPDecompOp : public OpKernel {
  public:
    GZIPDecompOp(OpKernelConstruction *ctx) : OpKernel(ctx) {};

    ~GZIPDecompOp() {
      core::ScopedUnref unref_pool(buffer_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buffer_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pool", &buffer_pool_));
      }

      const Tensor *data_handle_t;
      OP_REQUIRES_OK(ctx, ctx->input("data_handle", &data_handle_t));
      auto data_handle_vec = data_handle_t->vec<string>();

      ResourceContainer<Data> *data_handle;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data_handle_vec(0),
                                                          data_handle_vec(1),
                                                          &data_handle));
      core::ScopedUnref data_unref(data_handle);
      {
        ResourceReleaser<Data> data_handle_releaser(*data_handle);
        ResourceContainer<Buffer> *decomp_output;
        OP_REQUIRES_OK(ctx, buffer_pool_->GetResource(&decomp_output));
        decomp_output->get()->reset();
        auto& buff = decomp_output->get()->get();
        auto data = data_handle->get()->data();
        auto data_size = data_handle->get()->size();
        buff.reserve(data_size);

        OP_REQUIRES_OK(ctx, decompressGZIP(data, data_size, buff));
        OP_REQUIRES_OK(ctx, decomp_output->allocate_output("uncomp_handle", ctx));
      }
    }
  private:
    ReferencePool<Buffer> *buffer_pool_ = nullptr;
  };
} // namespace tensorflow {
