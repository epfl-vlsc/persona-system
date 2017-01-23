#include <utility>
#include <atomic>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool_op.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"
#include "bwa_reads.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("BWAAssembler"), no_meta_op_name("NoMetaBWAAssembler"), bwa_read_pool("BWAReadPool");
  }

  REGISTER_OP(op_name.c_str())
  .Input("bwa_read_pool: Ref(string)")
  .Input("base_handle: string")
  .Input("qual_handle: string")
  .Input("meta_handle: string")
  .Input("num_records: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
      })
  .Output("bwa_read_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

  REGISTER_OP(no_meta_op_name.c_str())
  .Input("bwa_read_pool: Ref(string)")
  .Input("base_handle: string")
  .Input("qual_handle: string")
  .Input("num_records: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(2));
      return Status::OK();
      })
  .Output("bwa_read_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

  REGISTER_REFERENCE_POOL(bwa_read_pool.c_str())
  .Doc(R"doc(
A pool specifically for bwa read resources.

Intended to be used for BWAAssembler
)doc");

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

      LOG(INFO) << "assembler outputting read resource!";
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

  class BWAAssemblerPoolOp : public ReferencePoolOp<BWAReadResource, ReadResource> {
  public:
    BWAAssemblerPoolOp(OpKernelConstruction *ctx) : ReferencePoolOp<BWAReadResource, ReadResource>(ctx) {}

  protected:
    unique_ptr<BWAReadResource> CreateObject() override {
      return unique_ptr<BWAReadResource>(new BWAReadResource());
    };
  };


  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), BWAAssemblerOp);
  REGISTER_KERNEL_BUILDER(Name(no_meta_op_name.c_str()).Device(DEVICE_CPU), NoMetaBWAAssemblerOp);
  REGISTER_KERNEL_BUILDER(Name(bwa_read_pool.c_str()).Device(DEVICE_CPU), BWAAssemblerPoolOp);
} // namespace tensorflow {
