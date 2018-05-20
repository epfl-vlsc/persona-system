
#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_executor.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {

// Defines a QueueOp, an abstract class for Queue construction ops.
class AlignmentExecutorOp : public ResourceOpKernel<AlignmentExecutor> {
 public:
  AlignmentExecutorOp(OpKernelConstruction* context) : ResourceOpKernel(context) {
    env_ = context->env();
    OP_REQUIRES_OK(context, context->GetAttr("num_threads", &num_threads_));
    OP_REQUIRES_OK(context, context->GetAttr("capacity", &capacity_));
  }

 private:
  Status CreateResource(AlignmentExecutor** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    LOG(INFO) << "Creating alignment resource with " << num_threads_ << " threads and "
      << capacity_ << " capacity";
    AlignmentExecutor* map = new AlignmentExecutor(env_, num_threads_, capacity_);
    *ret = map;
    return Status::OK();
  }

  Env* env_ = nullptr;
  int num_threads_ = 1;
  int capacity_ = 100;
  
  TF_DISALLOW_COPY_AND_ASSIGN(AlignmentExecutorOp);

};

REGISTER_KERNEL_BUILDER(Name("AlignmentExecutor").Device(DEVICE_CPU), AlignmentExecutorOp);

}
