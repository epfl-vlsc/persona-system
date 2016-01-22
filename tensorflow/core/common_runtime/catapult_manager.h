#ifndef TENSORFLOW_COMMON_RUNTIME_CATAPULT_MANAGER
#define TENSORFLOW_COMMON_RUNTIME_CATAPULT_MANAGER

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/fpga_manager.h"

namespace tensorflow {

class CatapultManager : public FPGAManagerBase {
    public:
        CatapultManager(const SessionOptions& options);
        ~CatapultManager() {};
        void FPGACompute(OpKernel* op_kernel, OpKernelContext* context) override;
        int FPGADeviceStatus () override { return device_status_; }
    private:
        string bitstreams_path_; // file path for catapult bitstreams
                                             // must be same names as opkernel->name()
        string current_bitstream_;
        int device_status_; // set to indicate fpga status
                            // <0 == bad, >= 0 == good
        TF_DISALLOW_COPY_AND_ASSIGN(CatapultManager);
};

}  // namespace tensorflow

#endif
