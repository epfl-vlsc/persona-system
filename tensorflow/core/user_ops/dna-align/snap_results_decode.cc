
#include "tensorflow/core/user_ops/dna-align/snap_results_decode.h"

namespace tensorflow {

MutableSnapResultsDecode::MutableSnapResultsDecode(Tensor* results_tensor) {
  if (results_tensor->dims() != 3) {
    LOG(INFO) << "Error: trying to create results decoder with tensor"
      << " not of dim 2";
  }
  if (results_tensor->dtype() != DT_INT64) {
    LOG(INFO) << "Error: trying to create results decoder with non "
      << "string type tensor.";
  }
  results_tensor_ = results_tensor;
  num_results_ = results_tensor->dim_size(0);
}

// set whether or not first result is primary
void MutableSnapResultsDecode::set_first_is_primary(int batch_index, bool value) {
  auto results = results_tensor_->tensor<int64, 3>();
  results(batch_index, 0, kFirstIsPrimary) = (int64) value;
}

bool MutableSnapResultsDecode::first_is_primary(int batch_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, 0, kFirstIsPrimary) != 0;
}

// set the number of results for the result in the given
// batch index
void MutableSnapResultsDecode::set_num_results(int batch_index, int value) {
  auto results = results_tensor_->tensor<int64, 3>();
  results(batch_index, 0, kNumResults) = value;
}

int64 MutableSnapResultsDecode::num_results(int batch_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, 0, kNumResults);
}

int64 MutableSnapResultsDecode::result_type(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kResultType);
}

void MutableSnapResultsDecode::set_result_type(int batch_index, int result_index, int64 value) {
  auto results = results_tensor_->tensor<int64, 3>();
  results(batch_index, result_index+1, kResultType) = value;
}

int64 MutableSnapResultsDecode::genome_location(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kGenomeLocation);
}

void MutableSnapResultsDecode::set_genome_location(int batch_index, int result_index, int64 value) {
  auto results = results_tensor_->tensor<int64, 3>();
  results(batch_index, result_index+1, kGenomeLocation) = value;
}

int64 MutableSnapResultsDecode::score(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kScore);
}

void MutableSnapResultsDecode::set_score(int batch_index, int result_index, int64 value) {
  auto results = results_tensor_->tensor<int64, 3>();
  results(batch_index, result_index+1, kScore) = value;
}

int64 MutableSnapResultsDecode::mapq(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kMapq);
}

void MutableSnapResultsDecode::set_mapq(int batch_index, int result_index, int64 value) {
  auto results = results_tensor_->tensor<int64, 3>();
  results(batch_index, result_index+1, kMapq) = value;
}

int64 MutableSnapResultsDecode::direction(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kDirection);
}

void MutableSnapResultsDecode::set_direction(int batch_index, int result_index, int64 value) {
  auto results = results_tensor_->tensor<int64, 3>();
  results(batch_index, result_index+1, kDirection) = value;
}


// Immutable 

SnapResultsDecode::SnapResultsDecode(const Tensor* results_tensor) {
  if (results_tensor->dims() != 3) {
    LOG(INFO) << "Error: trying to create results decoder with tensor"
      << " not of dim 2";
  }
  if (results_tensor->dtype() != DT_INT64) {
    LOG(INFO) << "Error: trying to create results decoder with non "
      << "string type tensor.";
  }
  results_tensor_ = results_tensor;
  num_results_ = results_tensor->dim_size(0);
}

bool SnapResultsDecode::first_is_primary(int batch_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, 0, kFirstIsPrimary) != 0;
}

int64 SnapResultsDecode::num_results(int batch_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, 0, kNumResults);
}

int64 SnapResultsDecode::result_type(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kResultType);
}

int64 SnapResultsDecode::genome_location(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kGenomeLocation);
}

int64 SnapResultsDecode::score(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kScore);
}

int64 SnapResultsDecode::mapq(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kMapq);
}

int64 SnapResultsDecode::direction(int batch_index, int result_index) {
  auto results = results_tensor_->tensor<int64, 3>();
  return results(batch_index, result_index+1, kDirection);
}

}

