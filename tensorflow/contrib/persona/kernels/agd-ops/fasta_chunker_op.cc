#include <memory>
#include <utility>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/agd-format/fasta_resource.h"
#include "tensorflow/contrib/persona/kernels/agd-format/fasta_chunker.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("FastaChunker"), pool_name("FastaReadPool");
    const TensorShape enqueue_shape_{{2}}, first_ord_shape_{};

    void custom_deleter(FastaResource::FileResource *f) {
      ResourceReleaser<Data> a(*f);
    }
  }


  class FastaChunkingOp : public OpKernel {
  public:
    FastaChunkingOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
    }

    ~FastaChunkingOp() {
      core::ScopedUnref queue_unref(queue_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      const Tensor *fasta_file_t;
      OP_REQUIRES_OK(ctx, ctx->input("fasta_file", &fasta_file_t));
      auto fasta_file_data = fasta_file_t->vec<string>();

      auto *rmgr = ctx->resource_manager();

      ResourceContainer<Data> *fasta_file;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(fasta_file_data(0), fasta_file_data(1), &fasta_file));
      core::ScopedUnref fasta_unref(fasta_file);
      shared_ptr<FastaResource::FileResource> file_data(fasta_file, custom_deleter);

      FastaChunker chunker(file_data, chunk_size_);

      // We can do this without a double move, but we have to release an unneeded resource at the end
      FastaResource fr;
      ResourceContainer<FastaResource> *fasta_resource;
      while (chunker.next_chunk(fr)) {
        OP_REQUIRES_OK(ctx, fasta_pool_->GetResource(&fasta_resource));
        *(fasta_resource->get()) = move(fr);
        OP_REQUIRES_OK(ctx, EnqueueFastaResource(ctx, fasta_resource));
      }
    }

  private:
    int64_t first_ordinal_ = 0;
    int chunk_size_;
    QueueInterface *queue_ = nullptr;
    ReferencePool<FastaResource> *fasta_pool_ = nullptr;

    Status EnqueueFastaResource(OpKernelContext *ctx, ResourceContainer<FastaResource> *fasta_resource) {
      QueueInterface::Tuple tuple;
      auto *fr = fasta_resource->get();
      auto num_records = fr->num_records();

      Tensor fasta_out, first_ord_out, num_recs_out;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, enqueue_shape_, &fasta_out));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, first_ord_shape_, &first_ord_out));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, first_ord_shape_, &num_recs_out));
      auto f_o = fasta_out.vec<string>();
      auto first_ord = first_ord_out.scalar<int64>();
      auto num_recs = num_recs_out.scalar<int64>();
      f_o(0) = fasta_resource->container();
      f_o(1) = fasta_resource->name();
      first_ord() = first_ordinal_;
      num_recs() = num_records;
      first_ordinal_ += num_records;
      tuple.push_back(fasta_out);
      tuple.push_back(first_ord_out);
      tuple.push_back(num_recs_out);
      TF_RETURN_IF_ERROR(queue_->ValidateTuple(tuple));
      Notification n;
      queue_->TryEnqueue(tuple, ctx, [&n]() { n.Notify(); });
      n.WaitForNotification();
      return Status::OK();
    }

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &queue_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "fasta_pool", &fasta_pool_));
      return Status::OK();
    }
  };

  class FastaPoolOp : public ReferencePoolOp<FastaResource, ReadResource> {
  public:
    FastaPoolOp(OpKernelConstruction *ctx) : ReferencePoolOp<FastaResource, ReadResource>(ctx) {}

  protected:
    unique_ptr<FastaResource> CreateObject() override {
      return unique_ptr<FastaResource>(new FastaResource());
    };
  private:
    TF_DISALLOW_COPY_AND_ASSIGN(FastaPoolOp);
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), FastaChunkingOp);
  REGISTER_KERNEL_BUILDER(Name(pool_name.c_str()).Device(DEVICE_CPU), FastaPoolOp);
} // namespace tensorflow {
