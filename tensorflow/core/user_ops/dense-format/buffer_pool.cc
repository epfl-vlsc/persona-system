#include "tensorflow/core/user_ops/object-pool/ref_pool_op.h"
#include "tensorflow/core/user_ops/dense-format/buffer.h"
#include "tensorflow/core/user_ops/dna-align/data.h"

namespace tensorflow {
  using namespace std;

  REGISTER_REFERENCE_POOL("BufferPool")
  .Attr("buffer_size: int64 = 20000000000") // 20 GigaBytes
  .Doc(R"doc(
Creates and initializes a pool containing char buffers of size `buffer_size` bytes
  )doc");

  class BufferPoolOp : public ReferencePoolOp<Buffer, Data> {
  public:
    BufferPoolOp(OpKernelConstruction* ctx) : ReferencePoolOp<Buffer, Data>(ctx) {
      using namespace errors;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("buffer_size", &buffer_size_));
      OP_REQUIRES(ctx, buffer_size_ > 0, InvalidArgument("BufferPoolOp requires buffer_size > 0 : ", buffer_size_));
    }

  protected:
    unique_ptr<Buffer> CreateObject() override {
      return unique_ptr<Buffer>(new Buffer(buffer_size_));
    }

  private:
    long long int buffer_size_;
  };

  REGISTER_KERNEL_BUILDER(Name("BufferPool").Device(DEVICE_CPU), BufferPoolOp);
} // namespace tensorflow {
