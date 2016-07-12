#include <utility>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool_op.h"
#include "fastq_iter.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("FASTQCreator"), pool_name("FASTQCreatorPool");
  };

  REGISTER_OP(op_name.c_str())
  .Input("pool_handle: Ref(string)")
  .Input("data_handle: string")
  .Output("fastq_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Takes an mmap'ed file (or from any other source) and wraps it in a
FASTQ_Iter object, which it passes downsteram
)doc");

  REGISTER_REFERENCE_POOL(pool_name.c_str())
  .Doc(R"doc(
A pool specifically design to be used for FASTQCreator

Recommended to make this unbound and manage the bounding with queues.
)doc");

  class FASTQCreatorOp : public OpKernel {
  public:
    FASTQCreatorOp(OpKernelConstruction *ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
      if (!fastq_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "pool_handle", &fastq_pool_));
      }

      ResourceContainer<Data> *fastq_file;
      const Tensor *input;
      OP_REQUIRES_OK(ctx, ctx->input("data_handle", &input));
      auto vec = input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(vec(0), vec(1), &fastq_file));
      core::ScopedUnref a(fastq_file);

      ResourceContainer<FASTQIterator> *fastq_handle;
      OP_REQUIRES_OK(ctx, fastq_pool_->GetResource(&fastq_handle));
      auto fastq = fastq_handle->get();
      // TODO need to assign a move constructor here!
      *fastq = move(FASTQIterator(fastq_file));
      OP_REQUIRES_OK(ctx, fastq_handle->allocate_output("fastq_handle", ctx));
    }
  private:
    ReferencePool<FASTQIterator> *fastq_pool_ = nullptr;
  };

  class FASTQCreatorPoolOp : public ReferencePoolOp<FASTQIterator, ReadResource> {
  public:

    FASTQCreatorPoolOp(OpKernelConstruction *ctx) : ReferencePoolOp<FASTQIterator, ReadResource>(ctx) {}

  protected:
    unique_ptr<FASTQIterator> CreateObject() override {
      return unique_ptr<FASTQIterator>(new FASTQIterator());
    }
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), FASTQCreatorOp);
  REGISTER_KERNEL_BUILDER(Name(pool_name.c_str()).Device(DEVICE_CPU), FASTQCreatorPoolOp);
} // namespace tensorflow {
