#include "tensorflow/core/common_runtime/fpga_device.h"
#include "tensorflow/core/common_runtime/catapult_manager.h"

// #include FPGA_API_CATAPULT

namespace tensorflow {
        
    CatapultManager::CatapultManager(const SessionOptions& options) : FPGAManagerBase(options, "catapult") {
        LOG(INFO) << "CREATING NEW CATAPULT MANAGER\n";

        // initialize FPGA API, test device health
        // device_handle_ = FPGACreateHandle() etc. 
        // if (device fail) LOG() <<  device_status_ = -1;
        
        bitstreams_path_ = options.config.fpga_options().bitstream_path();
        current_bitstream_ = "";
        device_status_ = 1;  // its all good for now
    };

    void CatapultManager::FPGACompute(OpKernel* op_kernel, OpKernelContext* context) {

        //LOG(INFO) << "FPGA Compute placeholder in CatapultManager class, executing on local CPU\n";
        // device specific initialization code if needed
        // op kernel will do op specific data marshalling and execution via API 
        //
        // if (current_bitstream_ == "" or current_bitstream != op_kernel->name())
        //      find bit stream
        //      open handle
        //      reconfigure
        //      close handle
        op_kernel->Compute(context);
    }
}

