#ifndef TENSORFLOW_COMMON_RUNTIME_FPGA_MANAGER_H_
#define TENSORFLOW_COMMON_RUNTIME_FPGA_MANAGER_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class FPGAManagerBase {
    public:
        FPGAManagerBase(const SessionOptions& options) {};
        ~FPGAManagerBase() {};
        virtual void FPGACompute(OpKernel* op_kernel, OpKernelContext* context);
    private:
        TF_DISALLOW_COPY_AND_ASSIGN(FPGAManagerBase);
};

}  // namespace tensorflow

#endif

