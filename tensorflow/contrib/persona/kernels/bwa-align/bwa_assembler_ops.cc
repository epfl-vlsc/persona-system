#include <utility>
#include <atomic>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "bwa_reads.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class BWAAssemblerOp : public OpKernel {
  public:
    BWAAssemblerOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      if (!drr_pool_) {
        OP_REQUIRES_OK(ctx, InitializePool(ctx));
      }

      const Tensor *base_data_t, *qual_data_t, *meta_data_t, *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("base_handle", &base_data_t));
      OP_REQUIRES_OK(ctx, ctx->input("qual_handle", &qual_data_t));
      OP_REQUIRES_OK(ctx, ctx->input("meta_handle", &meta_data_t));
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));

      auto bd = base_data_t->vec<string>();
      auto qd = qual_data_t->vec<string>();
      auto md = meta_data_t->vec<string>();
      auto num_records = num_records_t->scalar<int32>()();

      ResourceContainer<Data> *base_data, *qual_data, *meta_data;
      auto rmgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, rmgr->Lookup(bd(0), bd(1), &base_data));
      OP_REQUIRES_OK(ctx, rmgr->Lookup(qd(0), qd(1), &qual_data));
      OP_REQUIRES_OK(ctx, rmgr->Lookup(md(0), md(1), &meta_data));

      core::ScopedUnref b_unref(base_data), q_unref(qual_data), m_unref(meta_data);

      ResourceContainer<BWAReadResource> *bwa_reads;
      OP_REQUIRES_OK(ctx, drr_pool_->GetResource(&bwa_reads));

      //auto dr = bwa_reads->get();

      bwa_reads->assign(new BWAReadResource(num_records, base_data, qual_data, meta_data));
      OP_REQUIRES_OK(ctx, bwa_reads->allocate_output("bwa_read_handle", ctx));
    }
  private:

    inline Status InitializePool(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "bwa_read_pool", &drr_pool_));
      return Status::OK();
    }

    ReferencePool<BWAReadResource> *drr_pool_ = nullptr;
  };

  class NoMetaBWAAssemblerOp : public OpKernel {
  public:
    NoMetaBWAAssemblerOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      if (!drr_pool_) {
        OP_REQUIRES_OK(ctx, InitializePool(ctx));
      }
      start = clock();

      const Tensor *base_data_t, *qual_data_t, *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("base_handle", &base_data_t));
      OP_REQUIRES_OK(ctx, ctx->input("qual_handle", &qual_data_t));
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));

      auto bd = base_data_t->vec<string>();
      auto qd = qual_data_t->vec<string>();
      auto num_records = num_records_t->scalar<int32>()();

      ResourceContainer<Data> *base_data, *qual_data;
      auto rmgr = ctx->resource_manager();
      OP_REQUIRES_OK(ctx, rmgr->Lookup(bd(0), bd(1), &base_data));
      OP_REQUIRES_OK(ctx, rmgr->Lookup(qd(0), qd(1), &qual_data));

      core::ScopedUnref b_unref(base_data), q_unref(qual_data);

      ResourceContainer<BWAReadResource> *bwa_reads;
      OP_REQUIRES_OK(ctx, drr_pool_->GetResource(&bwa_reads));

      //auto dr = bwa_reads->get();

      //LOG(INFO) << "assembler outputting read resource!";
      bwa_reads->assign(new BWAReadResource(num_records, base_data, qual_data));
      OP_REQUIRES_OK(ctx, bwa_reads->allocate_output("bwa_read_handle", ctx));
    }
  private:
    clock_t start;

    inline Status InitializePool(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "bwa_read_pool", &drr_pool_));
      return Status::OK();
    }

    ReferencePool<BWAReadResource> *drr_pool_ = nullptr;
  };

  class BWAAssemblerPoolOp : public ReferencePoolOp<BWAReadResource, BWAReadResource> {
  public:
    BWAAssemblerPoolOp(OpKernelConstruction *ctx) : ReferencePoolOp<BWAReadResource, BWAReadResource>(ctx) {}

  protected:
    unique_ptr<BWAReadResource> CreateObject() override {
      return unique_ptr<BWAReadResource>(new BWAReadResource());
    };
  private:
    TF_DISALLOW_COPY_AND_ASSIGN(BWAAssemblerPoolOp);
  };


  REGISTER_KERNEL_BUILDER(Name("BWAAssembler").Device(DEVICE_CPU), BWAAssemblerOp);
  REGISTER_KERNEL_BUILDER(Name("NoMetaBWAAssembler").Device(DEVICE_CPU), NoMetaBWAAssemblerOp);
  REGISTER_KERNEL_BUILDER(Name("BWAReadPool").Device(DEVICE_CPU), BWAAssemblerPoolOp);
} // namespace tensorflow {
