#ifndef TENSORFLOW_COMMON_RUNTIME_HARP_MANAGER
#define TENSORFLOW_COMMON_RUNTIME_HARP_MANAGER

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/common_runtime/fpga_manager.h"

namespace tensorflow {

class HarpManager : public FPGAManagerBase {
    public:
        HarpManager(const SessionOptions& options);
        ~HarpManager() {};
        void FPGACompute(OpKernel* op_kernel, OpKernelContext* context) override;
        int FPGADeviceStatus() override { return -1; }  
    private:
        TF_DISALLOW_COPY_AND_ASSIGN(HarpManager);
};

}  // namespace tensorflow

#endif
