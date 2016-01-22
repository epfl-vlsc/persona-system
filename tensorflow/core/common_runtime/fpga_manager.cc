#include "tensorflow/core/common_runtime/fpga_device.h"
#include "tensorflow/core/common_runtime/fpga_manager.h"

namespace tensorflow {
    void FPGAManagerBase::FPGACompute(OpKernel* op_kernel, OpKernelContext* context) {

        LOG(FATAL) << "FPGAManagerBase class should never be instantiated!\n";
        //op_kernel->Compute(context);
    }
}

