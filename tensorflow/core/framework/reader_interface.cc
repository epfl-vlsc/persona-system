#include "reader_interface.h"

namespace tensorflow {
  Status ReaderInterface::Initialize(QueueInterface *queue, OpKernelContext *context)
  {
    if (needs_init_) {
      needs_init_ = false;
      return InitializeOnce(queue, context);
    }
    return Status::OK();
  }

  Status ReaderInterface::InitializeOnce(QueueInterface *queue, OpKernelContext *context)
  {
    return Status::OK();
  }
} // namespace tensorflow {
