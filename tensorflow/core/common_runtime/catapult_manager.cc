#include "tensorflow/core/common_runtime/fpga_device.h"
#include "tensorflow/core/common_runtime/catapult_manager.h"
#include <iostream>

namespace tensorflow {
    void CatapultManager::FPGACompute(OpKernel* op_kernel, OpKernelContext* context) {

        std::cout << "FPGA Compute placeholder in CatapultManager class, executing on local CPU\n";
        op_kernel->Compute(context);
    }
}

