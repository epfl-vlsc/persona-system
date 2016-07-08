
#include "tensorflow/core/user_ops/dna-align/snap_read_decode.h"

namespace tensorflow {

using namespace std;
SnapReadDecode::SnapReadDecode(const Tensor* read_tensor) {
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

const char* SnapReadDecode::bases(int batch_index) const {
  auto reads = read_tensor_->matrix<string>();
  return reads(kBases, batch_index).c_str();
}
const char* SnapReadDecode::qualities(int batch_index) const {
  auto reads = read_tensor_->matrix<string>();
  return reads(kQualities, batch_index).c_str();
}
const char* SnapReadDecode::metadata(int batch_index) const {
  auto reads = read_tensor_->matrix<string>();
  return reads(kMetadata, batch_index).c_str();
}
int SnapReadDecode::metadata_len(int batch_index) const {
  auto reads = read_tensor_->matrix<string>();
  return reads(kMetadata, batch_index).length();
}
int SnapReadDecode::bases_len(int batch_index) const {
  auto reads = read_tensor_->matrix<string>();
  return reads(kBases, batch_index).length();
}

// mutable
MutableSnapReadDecode::MutableSnapReadDecode(Tensor* read_tensor) {
  if (read_tensor->dims() != 2) {
    LOG(WARNING) << "Error: trying to create read decoder with tensor"
      << " not of dim 2";
  }
  if (read_tensor->dim_size(0) != 3) {
    LOG(WARNING) << "Error: need read tensor of dim-0 size of 3, but got "
              << read_tensor->dim_size(0);
  }
  if (read_tensor->dtype() != DT_STRING) {
    LOG(WARNING) << "Error: trying to create read decoder with non "
      << "string type tensor.";
  }
  read_tensor_ = read_tensor;
}

const char* MutableSnapReadDecode::bases(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(kBases, batch_index).c_str();
}
void MutableSnapReadDecode::set_bases(int batch_index, const string& bases) {
  auto reads = read_tensor_->matrix<string>();
  reads(kBases, batch_index) = bases;
}
const char* MutableSnapReadDecode::qualities(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(kQualities, batch_index).c_str();
}
void MutableSnapReadDecode::set_qualities(int batch_index, const string& quals) {
  auto reads = read_tensor_->matrix<string>();
  reads(kQualities, batch_index) = quals;
}
const char* MutableSnapReadDecode::metadata(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(kMetadata, batch_index).c_str();
}
void MutableSnapReadDecode::set_metadata(int batch_index, const string& meta) {
  auto reads = read_tensor_->matrix<string>();
  reads(kMetadata, batch_index) = meta;
}
int MutableSnapReadDecode::metadata_len(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(kMetadata, batch_index).length();
}
int MutableSnapReadDecode::bases_len(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(kBases, batch_index).length();
}

}

