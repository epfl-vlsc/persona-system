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
        CatapultManager(const SessionOptions& options) : FPGAManagerBase(options) {LOG(INFO) << "CREATING NEW CATAPULT MANAGER\n";};
        ~CatapultManager() {};
        void FPGACompute(OpKernel* op_kernel, OpKernelContext* context) override;
    private:
        TF_DISALLOW_COPY_AND_ASSIGN(CatapultManager);
};

}  // namespace tensorflow

#endif
