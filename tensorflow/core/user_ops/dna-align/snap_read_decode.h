
#ifndef TENSORFLOW_USER_OPS_SNAP_READ_DECODE_H_
#define TENSORFLOW_USER_OPS_SNAP_READ_DECODE_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include <string>

namespace tensorflow {

class SnapReadDecode {

  public:
    SnapReadDecode(const Tensor* read_tensor);

    const char* bases(int batch_index) const;

    const char* qualities(int batch_index) const;

    const char* metadata(int batch_index) const;

    int metadata_len(int batch_index) const;

    int bases_len(int batch_index) const;

    size_t size() const { return read_tensor_->dim_size(1); }

  private:
    static const int kBases = 0;
    static const int kQualities = 1;
    static const int kMetadata = 2;
    const Tensor* read_tensor_;
};

class MutableSnapReadDecode {

  public:
    MutableSnapReadDecode(Tensor* read_tensor);

    const char* bases(int batch_index);

    void set_bases(int batch_index, const std::string& bases);

    const char* qualities(int batch_index);

    void set_qualities(int batch_index, const std::string& quals);

    const char* metadata(int batch_index);

    void set_metadata(int batch_index, const std::string& meta);

    int metadata_len(int batch_index);

    int bases_len(int batch_index);

    size_t size() { return read_tensor_->dim_size(1); }

  private:
    static const int kBases = 0;
    static const int kQualities = 1;
    static const int kMetadata = 2;
    Tensor* read_tensor_;
};
}

#endif
