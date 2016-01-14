// Register a factory that provides FPGA devices.
#include "tensorflow/core/common_runtime/fpga_device.h"

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class FPGADeviceFactory : public DeviceFactory {
 public:
  void CreateDevices(const SessionOptions& options, const string& name_prefix,
                     std::vector<Device*>* devices) override {
    
    int n = 0;  // there should only be one FPGA for now
    // TODO: platfrom specific code to detect available devices
    auto iter = options.config.device_count().find("FPGA");
    if (iter != options.config.device_count().end()) {
      n = iter->second;
    }

    if (n && !options.config.has_fpga_options()) {
        VLOG(2) << "WARNING: Attempting to create an FPGA Device, but "
                << "no FPGA Options were specified in SessionOptions.\n";
        return;
    }
      
    for (int i = 0; i < n; i++) {
      string name = strings::StrCat(name_prefix, "/fpga:", i);
      devices->push_back(new FPGADevice(options, name, Bytes(256 << 20),
                                              BUS_ANY, cpu_allocator()));
    }
  }
};
REGISTER_LOCAL_DEVICE_FACTORY("FPGA", FPGADeviceFactory);

}  // namespace tensorflow
