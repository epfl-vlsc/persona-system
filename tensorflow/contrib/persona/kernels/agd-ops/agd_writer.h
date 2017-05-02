#pragma once

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"

namespace tensorflow {
  class AGDWriterBase : public OpKernel {
  public:
    AGDWriterBase(OpKernelConstruction *ctx);
  protected:
    Status SetOutputKey(OpKernelContext* ctx, const string &key);
    Status SetHeaderValues(OpKernelContext* ctx);

    format::FileHeader header_;
  private:
    std::string record_id_;
  };
} // namespace tensorflow {
