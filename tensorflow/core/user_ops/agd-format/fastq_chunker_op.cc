#include <memory>
#include <utility>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool_op.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "fastq_resource.h"
#include "fastq_chunker.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("FastqChunker"), pool_name("FastqReadPool");
    const TensorShape enqueue_shape_{{2}}, first_ord_shape_{};
  }

  REGISTER_OP(op_name.c_str())
  .Attr("chunk_size: int >= 1")
  .Input("queue_handle: resource")
  .Input("fastq_file: string") // TODO change this to resource when you update the op
  .Input("fastq_pool: Ref(string)")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
      using namespace shape_inference;

      ShapeHandle fastq_file;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &fastq_file));
      auto dim_handle = c->Dim(fastq_file, 0);
      auto fastq_dim = c->Value(dim_handle);
      if (fastq_dim != 2) {
        return Internal("fastq_file requires 2-dimensional vector");
      }

      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &fastq_file));
      dim_handle = c->Dim(fastq_file, 0);
      fastq_dim = c->Value(dim_handle);
      if (fastq_dim != 2) {
        return Internal("fastq_pool requires 2-dimensional vector");
      }

      return Status::OK();
    })
  .Doc(R"doc(

)doc");

  REGISTER_REFERENCE_POOL(pool_name.c_str())
  .Doc(R"doc(
A pool to manage FastqReadResource objects
)doc");

  class FastqChunkingOp : public OpKernel {
  public:
    FastqChunkingOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
    }

    ~FastqChunkingOp() {
      core::ScopedUnref queue_unref(queue_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      const Tensor *fastq_file_t;
      OP_REQUIRES_OK(ctx, ctx->input("fastq_file", &fastq_file_t));
      auto fastq_file_data = fastq_file_t->vec<string>();

      auto *rmgr = ctx->resource_manager();

      ResourceContainer<Data> *fastq_file;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(fastq_file_data(0), fastq_file_data(1), &fastq_file));
      core::ScopedUnref fastq_unref(fastq_file);
      shared_ptr<FastqResource::FileResource> file_data(fastq_file);

      FastqChunker chunker(file_data, chunk_size_);

      // We can do this without a double move, but we have to release an unneeded resource at the end
      FastqResource fr;
      ResourceContainer<FastqResource> *fastq_resource;
      while (chunker.next_chunk(fr)) {
        OP_REQUIRES_OK(ctx, fastq_pool_->GetResource(&fastq_resource));
        *(fastq_resource->get()) = move(fr);
        OP_REQUIRES_OK(ctx, EnqueueFastqResource(ctx, fastq_resource));
      }
    }

  private:
    int64_t first_ordinal_ = 0;
    int chunk_size_;
    QueueInterface *queue_ = nullptr;
    ReferencePool<FastqResource> *fastq_pool_ = nullptr;

    Status EnqueueFastqResource(OpKernelContext *ctx, ResourceContainer<FastqResource> *fastq_resource) {
      QueueInterface::Tuple tuple;
      auto *fr = fastq_resource->get();
      auto num_records = fr->num_records();

      Tensor fastq_out, first_ord_out, num_recs_out;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, enqueue_shape_, &fastq_out));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, first_ord_shape_, &first_ord_out));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, first_ord_shape_, &num_recs_out));
      auto f_o = fastq_out.vec<string>();
      auto first_ord = first_ord_out.scalar<int64>();
      auto num_recs = num_recs_out.scalar<int64>();
      f_o(0) = fastq_resource->container();
      f_o(1) = fastq_resource->name();
      first_ord() = first_ordinal_;
      num_recs() = num_records;
      first_ordinal_ += num_records;
      tuple.push_back(fastq_out);
      tuple.push_back(first_ord_out);
      tuple.push_back(num_recs_out);
      TF_RETURN_IF_ERROR(queue_->ValidateTuple(tuple));
      queue_->TryEnqueue(tuple, ctx, [](){});
      return Status::OK();
    }

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &queue_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "fastq_pool", &fastq_pool_));
      return Status::OK();
    }
  };

  class FastqPoolOp : public ReferencePoolOp<FastqResource, ReadResource> {
  public:
    FastqPoolOp(OpKernelConstruction *ctx) : ReferencePoolOp<FastqResource, ReadResource>(ctx) {}

  protected:
    unique_ptr<FastqResource> CreateObject() override {
      return unique_ptr<FastqResource>(new FastqResource());
    };
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), FastqChunkingOp);
  REGISTER_KERNEL_BUILDER(Name(pool_name.c_str()).Device(DEVICE_CPU), FastqPoolOp);
} // namespace tensorflow {
