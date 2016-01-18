#include "tensorflow/core/common_runtime/fpga_device.h"
#include "tensorflow/core/common_runtime/fpga_manager.h"
#include <iostream>

namespace tensorflow {
    void FPGAManagerBase::FPGACompute(OpKernel* op_kernel, OpKernelContext* context) {

        std::cout << "FPGA Compute placeholder in FPGAManagerBase class, executing on local CPU\n";
        op_kernel->Compute(context);
    }
}

