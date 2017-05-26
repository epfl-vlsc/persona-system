
// executor resource and provider op for snap paired aligner
#include <memory>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/GenomeIndex.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "tensorflow/contrib/persona/kernels/snap-align/SnapAlignerWrapper.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/snap-align/paired_executor.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  // the op that provides the executor via the resource manager
  class SnapPairedExecutorOp : public OpKernel {
  public:
    typedef BasicContainer<SnapPairedExecutor> ExecutorContainer;

    SnapPairedExecutorOp(OpKernelConstruction* context)
            : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads_));
      OP_REQUIRES_OK(context, context->GetAttr("work_queue_size", &capacity_));
      OP_REQUIRES_OK(context,
                     context->allocate_persistent(DT_STRING, TensorShape({ 2 }),
                                                  &executor_handle_, nullptr));
    }

    void Compute(OpKernelContext* ctx) override {
      mutex_lock l(mu_);
      if (!options_resource_)
        OP_REQUIRES_OK(ctx, InitHandles(ctx));
      if (!executor_handle_set_) {
        OP_REQUIRES_OK(ctx, SetExecutorHandle(ctx));
      }
      ctx->set_output_ref(0, &mu_, executor_handle_.AccessTensor(ctx));
    }

    ~SnapPairedExecutorOp() override {
      // If the genome object was not shared, delete it.
      if (executor_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
        TF_CHECK_OK(cinfo_.resource_manager()->Delete<ExecutorContainer>(
                cinfo_.container(), cinfo_.name()));
      }
    }

  private:
    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "options_handle", &options_resource_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "genome_handle", &index_resource_));
      TF_RETURN_IF_ERROR(snap_wrapper::init());

      return Status::OK();
    }

    Status SetExecutorHandle(OpKernelContext* ctx) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
      ExecutorContainer* new_executor;

      auto creator = [this, ctx](ExecutorContainer** executor) {
        LOG(INFO) << "creating snap paired executor";
        unique_ptr<SnapPairedExecutor> value(new SnapPairedExecutor(ctx->env(), index_resource_->get(),
                                                                    options_resource_->get(),
                                                                    num_threads_, capacity_));
        *executor = new ExecutorContainer(move(value));
        return Status::OK();
      };

      TF_RETURN_IF_ERROR(
              cinfo_.resource_manager()->LookupOrCreate<ExecutorContainer>(
                      cinfo_.container(), cinfo_.name(), &new_executor, creator));

      auto h = executor_handle_.AccessTensor(ctx)->flat<string>();
      h(0) = cinfo_.container();
      h(1) = cinfo_.name();
      executor_handle_set_ = true;
      return Status::OK();
    }

    mutex mu_;
    ContainerInfo cinfo_;
    int num_threads_, capacity_;
    BasicContainer<GenomeIndex> *index_resource_ = nullptr;
    BasicContainer<PairedAlignerOptions>* options_resource_ = nullptr;
    PersistentTensor executor_handle_ GUARDED_BY(mu_);
    volatile bool executor_handle_set_ = false GUARDED_BY(mu_);
    TF_DISALLOW_COPY_AND_ASSIGN(SnapPairedExecutorOp);
  };

  REGISTER_KERNEL_BUILDER(Name("SnapPairedExecutor").Device(DEVICE_CPU), SnapPairedExecutorOp);
}  // namespace tensorflow
