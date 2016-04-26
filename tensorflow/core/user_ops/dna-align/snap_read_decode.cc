
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

const char* SnapReadDecode::bases(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kBases).c_str();
}
const char* SnapReadDecode::qualities(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kQualities).c_str();
}
const char* SnapReadDecode::metadata(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kMetadata).c_str();
}
int SnapReadDecode::metadata_len(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kMetadata).length();
}
int SnapReadDecode::bases_len(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kBases).length();
}

// mutable
MutableSnapReadDecode::MutableSnapReadDecode(Tensor* read_tensor) {
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

const char* MutableSnapReadDecode::bases(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kBases).c_str();
}
void MutableSnapReadDecode::set_bases(int batch_index, string& bases) {
  auto reads = read_tensor_->matrix<string>();
  reads(batch_index, kBases) = bases;
}
const char* MutableSnapReadDecode::qualities(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kQualities).c_str();
}
void MutableSnapReadDecode::set_qualities(int batch_index, string& quals) {
  auto reads = read_tensor_->matrix<string>();
  reads(batch_index, kQualities) = quals;
}
const char* MutableSnapReadDecode::metadata(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kMetadata).c_str();
}
void MutableSnapReadDecode::set_metadata(int batch_index, string& meta) {
  auto reads = read_tensor_->matrix<string>();
  reads(batch_index, kMetadata) = meta;
}
int MutableSnapReadDecode::metadata_len(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kMetadata).length();
}
int MutableSnapReadDecode::bases_len(int batch_index) {
  auto reads = read_tensor_->matrix<string>();
  return reads(batch_index, kBases).length();
}

}

