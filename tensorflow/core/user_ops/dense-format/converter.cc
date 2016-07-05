#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "compression.h"
#include <string>

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
  .Input("chunk_buffer_pool: string")
  .Output("chunk_buffer_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
An op to convert a FASTQ file into multiple chunks
)doc");

  class DenseConverterOp : public OpKernel {
  public:
    DenseConverterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compress", &compress_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
      OP_REQUIRES(ctx, chunk_size_ > 0,
                  InvalidArgument("Chunk size (", chunk_size_, ") must be strictly >0"));
    }

    void Compute(OpKernelContext* ctx) override {
    }

  private:
    bool compress_;
    int chunk_size_;
  };
} // namespace tensorflow {
