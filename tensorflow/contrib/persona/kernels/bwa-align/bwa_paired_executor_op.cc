
// executor resource and provider op for snap single aligner
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>
#include <utility>
#include <chrono>
#include <atomic>
#include <vector>
#include <thread>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/bwa-align/bwa_paired_executor.h"
#include "tensorflow/contrib/persona/kernels/bwa-align/bwa_wrapper.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;


  // the op that provides the executor via the resource manager
  class BWAPairedExecutorOp : public OpKernel {
  public:
    typedef BasicContainer<BWAPairedExecutor> ExecutorContainer;

    BWAPairedExecutorOp(OpKernelConstruction* context)
            : OpKernel(context), executor_handle_set_(false) {
      OP_REQUIRES_OK(context, context->GetAttr("max_secondary", &max_secondary_));
      OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads_));
      OP_REQUIRES_OK(context, context->GetAttr("work_queue_size", &capacity_));
      OP_REQUIRES_OK(context, context->GetAttr("max_read_size", &max_read_size_));
      OP_REQUIRES_OK(context, context->GetAttr("thread_ratio", &thread_ratio_));
      OP_REQUIRES_OK(context,
                     context->allocate_persistent(DT_STRING, TensorShape({ 2 }),
                                                  &executor_handle_, nullptr));
    }

    void Compute(OpKernelContext* ctx) override {
      mutex_lock l(mu_);
      if (!options_resource_)
        OP_REQUIRES_OK(ctx, InitHandles(ctx));
      if (!executor_handle_set_) {
        OP_REQUIRES_OK(ctx, InitHandles(ctx));
        OP_REQUIRES_OK(ctx, SetExecutorHandle(ctx));
      }
      ctx->set_output_ref(0, &mu_, executor_handle_.AccessTensor(ctx));
    }

    ~BWAPairedExecutorOp() override {
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
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "index_handle", &index_resource_));

      return Status::OK();
    }

    Status SetExecutorHandle(OpKernelContext* ctx) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
      ExecutorContainer* new_executor;

      auto creator = [this, ctx](ExecutorContainer** executor) {
        LOG(INFO) << "creating snap single executor";
        unique_ptr<BWAPairedExecutor> value(new BWAPairedExecutor(ctx->env(), index_resource_->get(),
                                                                    options_resource_->get(), max_secondary_,
                                                                    num_threads_, capacity_, max_read_size_,
                                                                    thread_ratio_));
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
    float thread_ratio_;
    int max_secondary_, num_threads_, capacity_, max_read_size_;
    BasicContainer<bwaidx_t> *index_resource_ = nullptr;
    BasicContainer<mem_opt_t>* options_resource_ = nullptr;
    PersistentTensor executor_handle_ GUARDED_BY(mu_);
    bool executor_handle_set_ GUARDED_BY(mu_);
  };

  REGISTER_KERNEL_BUILDER(Name("BWAPairedExecutor").Device(DEVICE_CPU), BWAPairedExecutorOp);
}  // namespace tensorflow
