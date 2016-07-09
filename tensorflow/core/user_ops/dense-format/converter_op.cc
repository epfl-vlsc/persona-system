#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "shared_mmap_file_resource.h"
#include "column_builder.h"
#include "compression.h"
#include "buffer.h"
#include "fastq_iter.h"
#include <string>
#include <cstdint>

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("DenseConverter");
  }

  REGISTER_OP(op_name.c_str())
  .Attr("compress: bool")
  .Attr("chunk_size: int")
  .Input("fastq_file_handle: string")
  .Input("chunk_buffer_pool: Ref(string)")
  .Output("base_handle: string")
  .Output("qual_handle: string")
  .Output("meta_handle: string")
  .Output("num_records: int32") // to be used by the metadata op to issue ordinals
  .SetIsStateful()
  .Doc(R"doc(
An op to convert a FASTQ file into multiple chunks

This will emit multiple outputs for a single input (given the chunk size)
and will not pad to the chunk size (no need).

This version is slow because it doesn't parallelize reading a single file and compressing,
but this is just for the utility than the speed at this point.
)doc");

  class DenseConverterOp : public OpKernel {
  public:
    DenseConverterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compress", &compress_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
      OP_REQUIRES(ctx, chunk_size_ > 0,
                  InvalidArgument("Chunk size (", chunk_size_, ") must be strictly >0"));
    }

    ~DenseConverterOp() override {
      core::ScopedUnref pool_unref(buf_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buf_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "chunk_buffer_pool", &buf_pool_));
      }

      if (needs_new_file_) {
        GetNewFile(ctx);
      }

      ResourceContainer<Buffer> *base_buf_ctr, *qual_buf_ctr, *meta_buf_ctr;
      OP_REQUIRES_OK(ctx, buf_pool_->GetResource(&base_buf_ctr));
      OP_REQUIRES_OK(ctx, buf_pool_->GetResource(&qual_buf_ctr));
      OP_REQUIRES_OK(ctx, buf_pool_->GetResource(&meta_buf_ctr));

      auto base_buf = base_buf_ctr->get();
      auto qual_buf = qual_buf_ctr->get();
      auto meta_buf = meta_buf_ctr->get();

      base_buf->reset();
      qual_buf->reset();
      meta_buf->reset();

      // these are all vector<char>
      auto &base = base_buf->get();
      auto &qual = qual_buf->get();
      auto &meta = meta_buf->get();

      const char* bases, *qualities, *metadata;
      size_t bases_length, qualities_length, metadata_length;

      Status status;
      int num_records = 0;
      for (; num_records < chunk_size_; ++num_records) {
        status = fastq_iter_.get_next_record(&bases, &bases_length,
                                             &qualities, &qualities_length,
                                             &metadata, &metadata_length);
        if (status.ok()) {
          qual_builder_.AppendString(qualities, qualities_length, qual);
          meta_builder_.AppendString(metadata, metadata_length, meta);
          base_builder_.AppendString(bases, bases_length, base);
        } else {
          if (IsResourceExhausted(status)) {
            ReleaseFile();
            break;
          } else {
            OP_REQUIRES_OK(ctx, status);
          }
        }
      }

      OP_REQUIRES_OK(ctx, base_buf_ctr->allocate_output("base_handle", ctx));
      OP_REQUIRES_OK(ctx, qual_buf_ctr->allocate_output("qual_handle", ctx));
      OP_REQUIRES_OK(ctx, meta_buf_ctr->allocate_output("meta_handle", ctx));
      Tensor *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records", TensorShape(), &num_records_t));
      num_records_t->scalar<int>()() = num_records;
    }

  private:

    void GetNewFile(OpKernelContext *ctx) {
      ResourceContainer<Data> *fastq_file;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "fastq_file_handle", &fastq_file));
      fastq_iter_ = FASTQIterator(fastq_file);
      needs_new_file_ = false;
    }

    inline void ReleaseFile() {
      fastq_iter_ = FASTQIterator(); // run the constructor to release
      needs_new_file_ = true;
    }

    bool compress_;
    int chunk_size_;
    // Keep it like this instead of <Data> so that this converter op can close when it's done
    // This keeps memory to a minimum
    ReferencePool<Buffer> *buf_pool_ = nullptr;
    bool needs_new_file_ = true;
    FASTQIterator fastq_iter_;
    StringColumnBuilder meta_builder_, qual_builder_;
    BaseColumnBuilder base_builder_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), DenseConverterOp);
} // namespace tensorflow {
