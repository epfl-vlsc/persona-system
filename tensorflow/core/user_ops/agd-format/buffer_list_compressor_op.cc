#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"

#include "buffer.h"
#include "buffer_list.h"
#include "compression.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("BufferListCompressor");
  }

  REGISTER_OP(op_name.c_str())
  .Attr("buffer_pool_size: int >= 1")
  .Input("buffer_pool: Ref(string)")
  .Input("buffer_list: string")
  .Output("compressed_buffers: string")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
      using namespace shape_inference;

      ShapeHandle input_data;
      for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 2, &input_data));
        auto dim_handle = c->Dim(input_data, 0);
        auto dim_value = c->Value(dim_handle);
        if (dim_value != 2) {
          return Internal(op_name, ": input ", i, " needs value 2, but has ", dim_value);
        }
      }

      int32 buffer_output_dim;
      TF_RETURN_IF_ERROR(c->GetAttr("buffer_pool_size", &buffer_output_dim));

      input_data = c->Matrix(buffer_output_dim, 2);
      c->set_output(0, input_data);
    })
  .Doc(R"doc(
Compresses the prepared buffer_list records and into individual buffers, and then outputs them
)doc");

  class BufferListCompressorOp  : public OpKernel {
  public:
    BufferListCompressorOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_pool_size", &list_dimension_));
      output_shape_ = TensorShape({list_dimension_, 2});
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

      for (int i = 0; i < list_dimension_; i++) {
        auto &buf_pair = (*buf_list)[i];
        Buffer *buf = output_buffers_[i]->get();
        OP_REQUIRES_OK(ctx, CompressBuffer(buf_pair, buf));
      }

      OP_REQUIRES_OK(ctx, SendOutput(ctx));
    }
  private:
    ReferencePool<Buffer> *buf_pool_ = nullptr;
    int32 list_dimension_;
    TensorShape output_shape_;
    vector<ResourceContainer<Buffer>*> output_buffers_;

    Status CompressBuffer(BufferPair &buf_pair, Buffer *buf) {
      AppendingGZIPCompressor compressor(buf); // destructor releases GZIP resources
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

    Status GetResources(OpKernelContext *ctx, ResourceContainer<BufferList> **buffer_list) {
      const Tensor *input_data_t;
      TF_RETURN_IF_ERROR(ctx->input("buffer_list", &input_data_t));
      auto input_data_v = input_data_t->vec<string>();
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(input_data_v(0), input_data_v(1),
                                                         buffer_list));
      auto bl_size = (*buffer_list)->get()->size();
      if (bl_size != list_dimension_) {
        return Internal("BufferListCompressor: got a buffer list of size ", bl_size, " when the attribute specificed dimension ", list_dimension_);
      }

      output_buffers_.clear();

      decltype(output_buffers_)::value_type buf_rsrc_cntr;
      for (decltype(bl_size) i = 0; i < bl_size; i++) {
        TF_RETURN_IF_ERROR(buf_pool_->GetResource(&buf_rsrc_cntr));
        auto *buf = buf_rsrc_cntr->get();
        buf->reset();
        output_buffers_.push_back(buf_rsrc_cntr);
      }

      return Status::OK();
    }

    Status SendOutput(OpKernelContext* ctx) {
      Tensor *out_t;
      TF_RETURN_IF_ERROR(ctx->allocate_output("compressed_buffers", output_shape_, &out_t));
      auto out_matrix = out_t->matrix<string>();

      for (int i = 0; i < list_dimension_; i++) {
        auto *buf_rsrc_cntr = output_buffers_[i];
        out_matrix(i, 0) = buf_rsrc_cntr->container();
        out_matrix(i, 1) = buf_rsrc_cntr->name();
      }

      return Status::OK();
    }
  };
  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), BufferListCompressorOp);
} // namespace tensorflow {
