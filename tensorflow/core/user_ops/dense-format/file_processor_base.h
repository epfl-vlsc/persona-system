#ifndef TENSORFLOW_CORE_USEROPS_DENSE_ALIGN_FILE_PROCESSOR_BASE_H_
#define TENSORFLOW_CORE_USEROPS_DENSE_ALIGN_FILE_PROCESSOR_BASE_H_

#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

  class FileProcessorBase : public OpKernel {
  public:

    FileProcessorBase(OpKernelConstruction *context);
    void Compute(OpKernelContext* ctx) final;

  protected:

    virtual Status ProcessFile(MemoryMappedFile &mmf, OpKernelContext *ctx) = 0;
  };

} // namespace tensorflow {

#endif
