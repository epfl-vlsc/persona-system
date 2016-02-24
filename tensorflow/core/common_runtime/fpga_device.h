#ifndef TENSORFLOW_COMMON_RUNTIME_FPGA_DEVICE_H_
#define TENSORFLOW_COMMON_RUNTIME_FPGA_DEVICE_H_

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/fpga_manager.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// CPU device implementation.
class FPGADevice : public LocalDevice {
 public:
  FPGADevice(const SessionOptions& options, const string& name,
                   Bytes memory_limit, BusAdjacency bus_adjacency,
                   Allocator* allocator);
  ~FPGADevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  Status Sync() override { return Status::OK(); }
  int FPGADeviceStatus() { return fpga_manager_->FPGADeviceStatus(); }
  string FPGADeviceType() { return fpga_manager_->FPGADeviceType(); }

 private:
  Allocator* allocator_;  // Not owned
  FPGAManagerBase* fpga_manager_;  // for fpga system-specific compute
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_FPGA_DEVICE_H_
