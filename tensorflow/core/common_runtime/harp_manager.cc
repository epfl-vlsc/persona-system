#include "tensorflow/core/common_runtime/fpga_device.h"
#include "tensorflow/core/common_runtime/harp_manager.h"
#include <iostream>

namespace tensorflow {
    void HarpManager::FPGACompute(OpKernel* op_kernel, OpKernelContext* context) {

        std::cout << "FPGA Compute placeholder in HarpManager class\n";
    }
}

