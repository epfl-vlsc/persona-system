//
// Created by Stuart Byma on 21/04/17.
//

#include <memory>
#include <utility>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/agd-format/fastq_resource.h"
#include "tensorflow/contrib/persona/kernels/agd-format/fastq_chunker.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  namespace {
    const string op_name("FastqInterleavedChunker");
    const TensorShape enqueue_shape_{{2}}, first_ord_shape_{};

    void custom_deleter(FastqResource::FileResource *f) {
      ResourceReleaser<Data> a(*f);
    }
  }


  class FastqInterleavedChunkingOp : public OpKernel {
  public:
    FastqInterleavedChunkingOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
      // Chunk size is in terms of paired records
    }

    ~FastqInterleavedChunkingOp() {
      core::ScopedUnref queue_unref(queue_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      const Tensor *fastq_file_0_t, *fastq_file_1_t;
      OP_REQUIRES_OK(ctx, ctx->input("fastq_file_0", &fastq_file_0_t));
      auto fastq_file_0_data = fastq_file_0_t->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->input("fastq_file_1", &fastq_file_1_t));
      auto fastq_file_1_data = fastq_file_1_t->vec<string>();

      auto *rmgr = ctx->resource_manager();

      ResourceContainer<Data> *fastq_file_0, *fastq_file_1;

      OP_REQUIRES_OK(ctx, rmgr->Lookup(fastq_file_0_data(0), fastq_file_0_data(1), &fastq_file_0));
      core::ScopedUnref fastq_unref(fastq_file_0);
      shared_ptr<FastqResource::FileResource> file_data_0(fastq_file_0, custom_deleter);

      OP_REQUIRES_OK(ctx, rmgr->Lookup(fastq_file_1_data(0), fastq_file_1_data(1), &fastq_file_1));
      core::ScopedUnref fastq_unref_1(fastq_file_1);
      shared_ptr<FastqResource::FileResource> file_data_1(fastq_file_1, custom_deleter);

      FastqChunker chunker_0(file_data_0, chunk_size_);
      FastqChunker chunker_1(file_data_1, chunk_size_);

      // We can do this without a double move, but we have to release an unneeded resource at the end
      FastqResource fr_0, fr_1;
      ResourceContainer<FastqResource> *fastq_resource_0, *fastq_resource_1;
      while (chunker_0.next_chunk(fr_0)) {
        OP_REQUIRES(ctx, chunker_1.next_chunk(fr_1), Internal("Paired FASTQ files did not have matching number of chunks"));

        OP_REQUIRES_OK(ctx, fastq_pool_->GetResource(&fastq_resource_0));
        *(fastq_resource_0->get()) = move(fr_0);

        OP_REQUIRES_OK(ctx, fastq_pool_->GetResource(&fastq_resource_1));
        *(fastq_resource_1->get()) = move(fr_1);

        OP_REQUIRES_OK(ctx, EnqueueFastqResource(ctx, fastq_resource_0, fastq_resource_1));
      }
      if (chunker_1.next_chunk(fr_1)) {
        LOG(WARNING) << "Paired fastq chunker: second file has more chunks than the first!";
      }
    }

  private:
    int64_t first_ordinal_ = 0;
    int chunk_size_;
    QueueInterface *queue_ = nullptr;
    ReferencePool<FastqResource> *fastq_pool_ = nullptr;

    Status EnqueueFastqResource(OpKernelContext *ctx, ResourceContainer<FastqResource> *fastq_resource_0,
                                ResourceContainer<FastqResource> *fastq_resource_1) {
      QueueInterface::Tuple tuple;
      auto *fr_0 = fastq_resource_0->get();
      auto *fr_1 = fastq_resource_1->get();
      if (fr_0->num_records() != fr_1->num_records())
        return Internal("Fastq paired resources did not having matching number of records.");
      auto num_records = fr_0->num_records() * 2;

      Tensor fastq_out_0, fastq_out_1, first_ord_out, num_recs_out;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, enqueue_shape_, &fastq_out_0));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, enqueue_shape_, &fastq_out_1));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, first_ord_shape_, &first_ord_out));
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT64, first_ord_shape_, &num_recs_out));
      auto f_o_0 = fastq_out_0.vec<string>();
      auto f_o_1 = fastq_out_1.vec<string>();
      auto first_ord = first_ord_out.scalar<int64>();
      auto num_recs = num_recs_out.scalar<int64>();
      f_o_0(0) = fastq_resource_0->container();
      f_o_0(1) = fastq_resource_0->name();
      f_o_1(0) = fastq_resource_1->container();
      f_o_1(1) = fastq_resource_1->name();
      first_ord() = first_ordinal_;
      num_recs() = num_records;
      first_ordinal_ += num_records;
      tuple.push_back(fastq_out_0);
      tuple.push_back(fastq_out_1);
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
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "fastq_pool", &fastq_pool_));
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), FastqInterleavedChunkingOp);
} // namespace tensorflow {
