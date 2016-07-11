#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool_op.h"
#include "dense_reads.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("DenseAssembler"), dense_read_pool("DenseReadPool");
  }

  REGISTER_OP(op_name.c_str())
  .Input("dense_read_pool: Ref(string)")
  .Input("base_handle: string")
  .Input("qual_handle: string")
  .Input("meta_handle: string")
  .Input("num_records: int32")
  .Output("dense_read_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Assembles all 3 fields (bases, qualities, and metadata) into a generic reader object
which is passed downstream for conversion / alignment.

Currently this op requires all 3 fields to be available.
If we need to only process a subset in the future, we must make a separate op.
)doc");

  REGISTER_REFERENCE_POOL(dense_read_pool.c_str())
  .Doc(R"doc(
A pool specifically for dense read resources.

Intended to be used for DenseAssembler
)doc");

  class DenseAssemblerOp : public OpKernel {
  public:
    DenseAssemblerOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext* ctx) override {
      if (!drr_pool_) {
        OP_REQUIRES_OK(ctx, InitializePool(ctx));
      }

      ResourceContainer<DenseReadResource> *dense_reads;
      OP_REQUIRES_OK(ctx, drr_pool_->GetResource(&dense_reads));

      ResourceContainer<Data> *base_data, *qual_data, *meta_data;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "base_handle", &base_data));
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "qual_handle", &qual_data));
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "meta_handle", &meta_data));
      core::ScopedUnref b_unref(base_data), q_unref(qual_data), m_unref(meta_data);

      const Tensor *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->scalar<int32>()();

      auto dr = dense_reads->get();
      *dr = DenseReadResource(num_records, base_data, qual_data, meta_data);
      OP_REQUIRES_OK(ctx, dense_reads->allocate_output("dense_read_handle", ctx));
    }
  private:

    Status InitializePool(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "dense_read_handle", &drr_pool_));
      return Status::OK();
    }

    ReferencePool<DenseReadResource> *drr_pool_ = nullptr;
  };

  class DenseAssemblerPoolOp : public ReferencePoolOp<DenseReadResource, ReadResource> {
  public:
    DenseAssemblerPoolOp(OpKernelConstruction *ctx) : ReferencePoolOp<DenseReadResource, ReadResource>(ctx) {}

  protected:
    unique_ptr<DenseReadResource> CreateObject() override {
      return unique_ptr<DenseReadResource>(new DenseReadResource());
    };
  };


  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), DenseAssemblerOp);
  REGISTER_KERNEL_BUILDER(Name(dense_read_pool.c_str()).Device(DEVICE_CPU), DenseAssemblerPoolOp);
} // namespace tensorflow {
