#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("BufferListCompressor");
  }

  class BufferListCompressorOp  : public OpKernel {
  public:
    BufferListCompressorOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    }

    ~BufferListCompressorOp() {
      core::ScopedUnref a(buf_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buf_pool_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      ResourceContainer<BufferList> *buffer_list_resource;
      OP_REQUIRES_OK(ctx, GetResources(ctx, &buffer_list_resource));
      ResourceReleaser<BufferList> blr_releaser(*buffer_list_resource);
      auto *buf_list = buffer_list_resource->get();
      auto num_chunks = buf_list->size();

      ResourceContainer<Buffer> *output_buffer;
      OP_REQUIRES_OK(ctx, buf_pool_->GetResource(&output_buffer));
      auto *buf = output_buffer->get();
      buf->reset();
      AppendingGZIPCompressor compressor(*buf);
      OP_REQUIRES_OK(ctx, compressor.init());

      for (size_t i = 0; i < num_chunks; i++) {
        auto &buf_pair = (*buf_list)[i];
        auto &index = buf_pair.index();
        OP_REQUIRES_OK(ctx, compressor.appendGZIP(index.data(), index.size()));
      }

      for (size_t i = 0; i < num_chunks; i++) {
        auto &buf_pair = (*buf_list)[i];
        auto &data = buf_pair.data();
        OP_REQUIRES_OK(ctx, compressor.appendGZIP(data.data(), data.size()));
      }

      buf_list->reset();

      OP_REQUIRES_OK(ctx, output_buffer->allocate_output("buffer", ctx));
    }
  private:
    ReferencePool<Buffer> *buf_pool_ = nullptr;

    Status Init(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pool", &buf_pool_));
      return Status::OK();
    }

    Status GetResources(OpKernelContext *ctx, ResourceContainer<BufferList> **buffer_list) {
      const Tensor *input_data_t;
      TF_RETURN_IF_ERROR(ctx->input("buffer_list", &input_data_t));
      auto input_data_v = input_data_t->vec<string>();
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(input_data_v(0), input_data_v(1),
                                                         buffer_list));
      return Status::OK();
    }
  };
  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), BufferListCompressorOp);
} // namespace tensorflow {
