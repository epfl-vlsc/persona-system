#include <utility>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool_op.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_reads.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("AGDAssembler"), no_meta_op_name("NoMetaAGDAssembler"), agd_read_pool("AGDReadPool");
  }

  class AGDAssemblerOp : public OpKernel {
  public:
    AGDAssemblerOp(OpKernelConstruction *context) : OpKernel(context) {}

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

      ResourceContainer<AGDReadResource> *agd_reads;
      OP_REQUIRES_OK(ctx, drr_pool_->GetResource(&agd_reads));

      auto dr = agd_reads->get();

      AGDReadResource a(num_records, base_data, qual_data, meta_data);
      *dr = move(a);
      OP_REQUIRES_OK(ctx, agd_reads->allocate_output("agd_read_handle", ctx));
    }
  private:

    inline Status InitializePool(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "agd_read_pool", &drr_pool_));
      return Status::OK();
    }

    ReferencePool<AGDReadResource> *drr_pool_ = nullptr;
  };

  class NoMetaAGDAssemblerOp : public OpKernel {
  public:
    NoMetaAGDAssemblerOp(OpKernelConstruction *context) : OpKernel(context) {}

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

      ResourceContainer<AGDReadResource> *agd_reads;
      OP_REQUIRES_OK(ctx, drr_pool_->GetResource(&agd_reads));

      auto dr = agd_reads->get();

      AGDReadResource a(num_records, base_data, qual_data);
      *dr = move(a);
      OP_REQUIRES_OK(ctx, agd_reads->allocate_output("agd_read_handle", ctx));
    }
  private:
    clock_t start;

    inline Status InitializePool(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "agd_read_pool", &drr_pool_));
      return Status::OK();
    }

    ReferencePool<AGDReadResource> *drr_pool_ = nullptr;
  };

  class AGDAssemblerPoolOp : public ReferencePoolOp<AGDReadResource, ReadResource> {
  public:
    AGDAssemblerPoolOp(OpKernelConstruction *ctx) : ReferencePoolOp<AGDReadResource, ReadResource>(ctx) {}

  protected:
    unique_ptr<AGDReadResource> CreateObject() override {
      return unique_ptr<AGDReadResource>(new AGDReadResource());
    };
  private:
    TF_DISALLOW_COPY_AND_ASSIGN(AGDAssemblerPoolOp);
  };


  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDAssemblerOp);
  REGISTER_KERNEL_BUILDER(Name(no_meta_op_name.c_str()).Device(DEVICE_CPU), NoMetaAGDAssemblerOp);
  REGISTER_KERNEL_BUILDER(Name(agd_read_pool.c_str()).Device(DEVICE_CPU), AGDAssemblerPoolOp);
} // namespace tensorflow {
