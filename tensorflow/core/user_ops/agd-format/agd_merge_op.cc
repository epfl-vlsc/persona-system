#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/queue_interface.h"

#include "format.h"
#include "compression.h"
#include "parser.h"
#include "util.h"
#include "buffer.h"

#include "tensorflow/core/user_ops/lttng/tracepoints.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("AGDMerge");
  }

  REGISTER_OP(op_name.c_str())
  .Attr("chunk_size: int32 >= 1")
  .Input("buffer_pool: Ref(string)")
  .Input("num_records: int32")
  .Input("base_handle: string")
  .Input("qual_handle: string")
  .Input("meta_handle: string")
  .Input("results_handle: string")
  .Input("output_buffer_queue_handle: Ref(string)")
  .Output("")
  .Doc(R"doc(
Merges multiple input chunks into chunks based on `chunk_size`
Only supports a single-stage of merging, i.e. this will not write out to an arbitrarily-large single chunk.

chunk_size: the size, in number of records, of the output chunks
num_records: vector of number of recrods
*_handles: matrix of processed handles
)doc");

  class AGDMergeOp : public OpKernel {
  public:
    AGDMergeOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
    }

    ~AGDMergeOp() {
      core::ScopedUnref queue_unref(queue_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!queue_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "output_buffer_queue_handle", &queue_));
      }

      const Tensor *num_records_t, *bases_t, *quals_t, *meta_t, *results_t;
      OP_REQUIRES_OK(ctx, ctx->input("base_handle", &bases_t));
      OP_REQUIRES_OK(ctx, ctx->input("qual_handle", &quals_t));
      OP_REQUIRES_OK(ctx, ctx->input("meta_handle", &meta_t));
      OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_t));
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->vec<int32>();
      auto base_handles = bases_t->matrix<string>();
      auto qual_handles = quals_t->matrix<string>();
      auto meta_handles = meta_t->matrix<string>();
      auto result_handles = results_t->matrix<string>();
    }
  private:
    QueueInterface *queue_ = nullptr;
    int chunk_size_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDMergeOp);
} // namespace tensorflow {
