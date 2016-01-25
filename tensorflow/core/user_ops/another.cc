#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/device_base.h"

// include FPGA APIs ?
using namespace tensorflow;

REGISTER_OP("Another")
    .Output("another: string");

class AnotherOp : public OpKernel {
 public:
  explicit AnotherOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    // get the FPGA device pointer so we know which API to use
    // TODO: find a better way, because this won't work if this kernel is 
    // registered with other devices, could use dynamic_cast
    DeviceBase * device = context->device();
	const DeviceBase::FpgaDeviceInfo* fpga_info = device->tensorflow_fpga_device_info();
    if (fpga_info->device_type == "catapult") {
        // placeholder crap
        // Create an output tensor
        Tensor* output_tensor = NULL;
        //LOG(INFO) << "SORT COMPUTE WAS CALLED!\n";
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, TensorShape(), &output_tensor));
        auto output = output_tensor->template scalar<string>();

        output() = "This is the test of the FPGA OP!\n";
    } else {
        LOG(FATAL) << "Op kernel FPGATest is not implemented for FPGA device \"" << 
            fpga_info->device_type << "\"\n";
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Another").Device(DEVICE_CPU), AnotherOp);
REGISTER_KERNEL_BUILDER(Name("Another").Device(DEVICE_FPGA), AnotherOp);

