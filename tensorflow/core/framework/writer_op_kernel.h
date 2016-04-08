
// Stuart Byma
// Mostly copied from reader_op_kernel.h


#ifndef TENSORFLOW_FRAMEWORK_WRITER_OP_KERNEL_H_
#define TENSORFLOW_FRAMEWORK_WRITER_OP_KERNEL_H_

#include <functional>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/writer_interface.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Implementation for ops providing a Writer.
class WriterOpKernel : public OpKernel {
 public:
  explicit WriterOpKernel(OpKernelConstruction* context);
  ~WriterOpKernel() override;

  void Compute(OpKernelContext* context) override;

  // Must be called by descendants before the first call to Compute()
  // (typically called during construction).  factory must return a
  // WriterInterface descendant allocated with new that WriterOpKernel
  // will take ownership of.
  void SetWriterFactory(std::function<WriterInterface*()> factory) {
    mutex_lock l(mu_);
    DCHECK(!have_handle_);
    factory_ = factory;
  }

 private:
  mutex mu_;
  bool have_handle_ GUARDED_BY(mu_);
  PersistentTensor handle_ GUARDED_BY(mu_);
  ContainerInfo cinfo_;
  std::function<WriterInterface*()> factory_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_WRITER_OP_KERNEL_H_
