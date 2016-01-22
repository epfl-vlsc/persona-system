#include "tensorflow/core/common_runtime/fpga_device.h"
#include "tensorflow/core/common_runtime/harp_manager.h"
#include <iostream>

namespace tensorflow {

    HarpManager::HarpManager(const SessionOptions& options) : FPGAManagerBase(options, "harp") {
        LOG(INFO) << "CREATING NEW HARP MANAGER\n";

        // initialize FPGA API, test device health
        // device_handle_ = FPGACreateHandle() etc. 
        // if (device fail) LOG() <<  device_status_ = -1;
        
    };
    void HarpManager::FPGACompute(OpKernel* op_kernel, OpKernelContext* context) {

        std::cout << "FPGA Compute placeholder in HarpManager class\n";
    }
}

