
#ifndef TENSORFLOW_USER_OPS_SNAP_READ_DECODE_H_
#define TENSORFLOW_USER_OPS_SNAP_READ_DECODE_H_

#include "Read.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

class SnapReadDecode {

  public:
    SnapReadDecode(Tensor* read_tensor) {
      if (read_tensor->dims() != 2) {
        LOG(INFO) << "Error: trying to create read decoder with tensor"
          << " not of dim 2";
      }
      if (read_tensor->dtype() != DT_STRING) {
        LOG(INFO) << "Error: trying to create read decoder with non "
          << "string type tensor.";
      }
      read_tensor_ = read_tensor;
    }

    char* bases(int batch_index) {
      auto reads = read_tensor_->matrix<string>();
      return reads(batch_index, kBases).c_str();
    }
    char* qualities(int batch_index) {
      auto reads = read_tensor_->matrix<string>();
      return reads(batch_index, kQualities).c_str();
    }
    char* metadata(int batch_index) {
      auto reads = read_tensor_->matrix<string>();
      return reads(batch_index, kMetadata).c_str();
    }
    int metadata_len(int batch_index) {
      auto reads = read_tensor_->matrix<string>();
      return reads(batch_index, kMetadata).length();
    }
    int bases_len(int batch_index) {
      auto reads = read_tensor_->matrix<string>();
      return reads(batch_index, kBases).length();
    }
  private:
    const int kBases = 0;
    const int kQualities = 1;
    const int kMetadata = 2;
    Tensor* read_tensor_;
};

}


#endif
