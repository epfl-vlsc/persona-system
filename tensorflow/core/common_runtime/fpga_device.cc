#include "tensorflow/core/common_runtime/fpga_device.h"
#include "tensorflow/core/common_runtime/harp_manager.h"
#include "tensorflow/core/common_runtime/catapult_manager.h"

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/public/session_options.h"
//#include <iostream>

namespace tensorflow {

FPGADevice::FPGADevice(const SessionOptions& options,
                                   const string& name, Bytes memory_limit,
                                   BusAdjacency bus_adjacency,
                                   Allocator* allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_FPGA, memory_limit, bus_adjacency),
                  allocator),
      allocator_(allocator) {

  // instantiate manager for FPGA system
  // TODO factoryize 
  FPGAOptions f_options = options.config.fpga_options();
  if (f_options.system_type() == "harp")
      fpga_manager_ = new HarpManager(options);
  else if (f_options.system_type() == "catapult")
      fpga_manager_ = new CatapultManager(options);
  else
      LOG(FATAL) << "Unrecognized FPGA system type \"" << f_options.system_type() << "\" \n";
          // initialize FPGA device and acquire lock on it?
          //std::cout << "OMG CREATING NEW FPGA DEVICE!!!\n";
        
}

FPGADevice::~FPGADevice() {
  delete fpga_manager_;
}

void FPGADevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
    //op_kernel->Compute(context);
    fpga_manager_->FPGACompute(op_kernel, context);
  } else {
    fpga_manager_->FPGACompute(op_kernel, context);
    //op_kernel->Compute(context);
  }
}

Allocator* FPGADevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_;
}


// same as threadpool device for now
Status FPGADevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }
  *tensor = parsed;
  return Status::OK();
}

}  // namespace tensorflow
