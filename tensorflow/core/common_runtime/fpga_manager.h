#ifndef TENSORFLOW_COMMON_RUNTIME_FPGA_MANAGER_H_
#define TENSORFLOW_COMMON_RUNTIME_FPGA_MANAGER_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

    //mostly an abstract class
class FPGAManagerBase {
    public:
        FPGAManagerBase(const SessionOptions& options, string deviceType) : device_type_(deviceType) {};
        ~FPGAManagerBase() {};
        virtual void FPGACompute(OpKernel* op_kernel, OpKernelContext* context) = 0;
        string FPGADeviceType() { return device_type_; }
        virtual int FPGADeviceStatus() = 0;
    private:
        string device_type_;  // "catapult" or "harp" for now
        TF_DISALLOW_COPY_AND_ASSIGN(FPGAManagerBase);
};

}  // namespace tensorflow

#endif

