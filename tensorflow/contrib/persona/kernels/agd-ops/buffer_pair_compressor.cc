#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("BufferPairCompressor");
  }


  class BufferPairCompressorOp  : public OpKernel {
  public:
    BufferPairCompressorOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pack", &pack_));
    }

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

      core::ScopedUnref a(buffer_pair_resource);
      {
        ResourceReleaser<BufferPair> blr_releaser(*buffer_pair_resource);
        auto *buf_pair = buffer_pair_resource->get();
        auto *buf = buffer_resource->get();

        if (pack_) {
          OP_REQUIRES_OK(ctx, CompactBases(*buf_pair));
        }

        OP_REQUIRES_OK(ctx, CompressBuffer(*buf_pair, buf));
      }
    }

  private:
    ReferencePool<Buffer> *buf_pool_ = nullptr;
    bool pack_ = false;
    vector<format::BinaryBases> compact_;

    Status CompactBases(BufferPair& bufpair) {
      const format::RelativeIndex* index = reinterpret_cast<const format::RelativeIndex*>(bufpair.index().data());
      auto index_size = bufpair.index().size();
      const auto data = bufpair.data().data();

      // resetting just changes the size to 0,
      // we will overwrite with new compacted bases
      bufpair.reset();

      uint64 current_offset = 0;
      for (size_t i = 0; i < index_size; i++) {
        auto entry_size = index[i];
        auto* base_data = &data[current_offset];
        Status s = format::IntoBases(base_data, entry_size, compact_);
        if (!s.ok()) {
          string test(base_data, entry_size);
          LOG(INFO) << "compacting: " << test;
        }
        TF_RETURN_IF_ERROR(s);
        TF_RETURN_IF_ERROR(AppendRecord(&compact_[0], compact_.size(), bufpair));
        current_offset += entry_size;
      }
      return Status::OK();
    }

    template <typename T>
    Status AppendRecord(const T* data, unsigned elements_t, BufferPair &bp) {
      auto &index = bp.index();
      auto &data_buf = bp.data();

      int64_t elements = elements_t * sizeof(T);
      if (elements > format::MAX_INDEX_SIZE) {
        return Internal("Record size in bytes (", elements, ") exceeds the maximum (", format::MAX_INDEX_SIZE, ")");
      }

      auto converted_size = static_cast<format::RelativeIndex>(elements);
      auto converted_data = reinterpret_cast<const char*>(data);

      TF_RETURN_IF_ERROR(index.AppendBuffer(reinterpret_cast<char*>(&converted_size), sizeof(converted_size)));
      TF_RETURN_IF_ERROR(data_buf.AppendBuffer(converted_data, converted_size));

      return Status::OK();
    }

    Status CompressBuffer(BufferPair &buf_pair, Buffer *buf) {
      AppendingGZIPCompressor compressor(*buf); // destructor releases GZIP resources
      TF_RETURN_IF_ERROR(compressor.init());
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

      return Status::OK();
    }

  };
  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), BufferPairCompressorOp);
} // namespace tensorflow {
