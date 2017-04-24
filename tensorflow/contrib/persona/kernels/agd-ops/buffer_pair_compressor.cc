#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("BufferPairCompressor");
  }


  class BufferPairCompressorOp  : public OpKernel {
  public:
    BufferPairCompressorOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

    ~BufferPairCompressorOp() {
      core::ScopedUnref a(buf_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buf_pool_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      ResourceContainer<BufferPair> *buffer_pair_resource;
      ResourceContainer<Buffer> *buffer_resource;
      OP_REQUIRES_OK(ctx, GetResources(ctx, &buffer_pair_resource, &buffer_resource));
      ResourceReleaser<BufferPair> blr_releaser(*buffer_pair_resource);
      auto *buf_pair = buffer_pair_resource->get();
      auto *buf = buffer_resource->get();

      OP_REQUIRES_OK(ctx, CompressBuffer(*buf_pair, buf));
    }

  private:
    ReferencePool<Buffer> *buf_pool_ = nullptr;

    Status CompressBuffer(BufferPair &buf_pair, Buffer *buf) {
      AppendingGZIPCompressor compressor(*buf); // destructor releases GZIP resources
      compressor.init();
      auto &index = buf_pair.index();
      TF_RETURN_IF_ERROR(compressor.appendGZIP(index.data(), index.size()));
      auto &data = buf_pair.data();
      TF_RETURN_IF_ERROR(compressor.appendGZIP(data.data(), data.size()));
      return Status::OK();
    }

    Status Init(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pool", &buf_pool_));
      return Status::OK();
    }

    Status GetResources(OpKernelContext *ctx, ResourceContainer<BufferPair> **buffer_pair, 
        ResourceContainer<Buffer> **buffer) {
      const Tensor *input_data_t;
      TF_RETURN_IF_ERROR(ctx->input("buffer_pair", &input_data_t));
      auto input_data_v = input_data_t->vec<string>();
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(input_data_v(0), input_data_v(1),
                                                         buffer_pair));

      TF_RETURN_IF_ERROR(buf_pool_->GetResource(buffer));
      (*buffer)->get()->reset();
     
      TF_RETURN_IF_ERROR((*buffer)->allocate_output("compressed_buffer", ctx));
      //LOG(INFO) << "compressed buffer: " << (*buffer)->container() << ", " << (*buffer)->name();

      return Status::OK();
    }

  };
  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), BufferPairCompressorOp);
} // namespace tensorflow {
