
#ifndef TENSORFLOW_USER_OPS_SNAP_READ_DECODE_H_
#define TENSORFLOW_USER_OPS_SNAP_READ_DECODE_H_

#include "Read.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

// results are held in a 3 dimensional tensor
// a results tensor holds a batch of results
// each batch entry is a set of results for one read, where index 0
// is special and holds whether the first result is primary, and the 
// number of results. 
// To the user however, results are still 0-indexed
class SnapResultsDecode {

  public:
    SnapesultsDecode(Tensor* results_tensor) {
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

    // helper to get appropriate shape for given batch size and 
    // secondary results size
    static TensorShape get_results_shape(int batch_size,
        int num_secondary_results) {
      return TensorShape({batch_size, num_secondary_results+1, 5});
    }

    // set whether or not first result is primary
    void set_first_is_primary(int batch_index, bool value) {
      auto results = results_tensor_->tensor<string, 3>();
      results(batch_index, 0, kFirstIsPrimary) = (int64) value;
    }
    
    bool first_is_primary(int batch_index) {
      auto results = results_tensor_->tensor<string, 3>();
      return results(batch_index, 0, kFirstIsPrimary) != 0;
    }
   
    // set the number of results for the result in the given
    // batch index
    void set_num_results(int batch_index, int value) {
      auto results = results_tensor_->tensor<string, 3>();
      results(batch_index, 0, kNumResults) = value;
    }

    int64 num_results(int batch_index) {
      auto results = results_tensor_->tensor<string, 3>();
      return results(batch_index, 0, kNumResults);
    }

    int64 result_type(int batch_index, int result_index) {
      auto results = results_tensor_->tensor<string, 3>();
      return results(batch_index, result_index+1, kResultType);
    }
    
    void set_result_type(int batch_index, int result_index, int64 value) {
      auto results = results_tensor_->tensor<string, 3>();
      results(batch_index, result_index+1, kResultType) = value;
    }
    
    int64 genome_location(int batch_index, int result_index) {
      auto results = results_tensor_->tensor<string, 3>();
      return results(batch_index, result_index+1, kGenomeLocation);
    }
    
    void set_genome_location(int batch_index, int result_index, int64 value) {
      auto results = results_tensor_->tensor<string, 3>();
      results(batch_index, result_index+1, kGenomeLocation) = value;
    }
    
    int64 score(int batch_index, int result_index) {
      auto results = results_tensor_->tensor<string, 3>();
      return results(batch_index, result_index+1, kScore);
    }
    
    void set_score(int batch_index, int result_index, int64 value) {
      auto results = results_tensor_->tensor<string, 3>();
      results(batch_index, result_index+1, kScore) = value;
    }
    
    int64 mapq(int batch_index, int result_index) {
      auto results = results_tensor_->tensor<string, 3>();
      return results(batch_index, result_index+1, kMapq);
    }
    
    void set_mapq(int batch_index, int result_index, int64 value) {
      auto results = results_tensor_->tensor<string, 3>();
      results(batch_index, result_index+1, kMapq) = value;
    }
    
    int64 mapq(int batch_index, int result_index) {
      auto results = results_tensor_->tensor<string, 3>();
      return results(batch_index, result_index+1, kMapq);
    }
    
    void set_mapq(int batch_index, int result_index, int64 value) {
      auto results = results_tensor_->tensor<string, 3>();
      results(batch_index, result_index+1, kMapq) = value;
    }
    
    int64 direction(int batch_index, int result_index) {
      auto results = results_tensor_->tensor<string, 3>();
      return results(batch_index, result_index+1, kDirection);
    }
    
    void set_direction(int batch_index, int result_index, int64 value) {
      auto results = results_tensor_->tensor<string, 3>();
      results(batch_index, result_index+1, kDirection) = value;
    }

  private:
    const int kFirstIsPrimary = 0;
    const int kNumResults = 1;
    const int kResultType = 0;
    const int kGenomeLocation = 1;
    const int kScore = 2;
    const int kMapq = 3;
    const int kDirection = 4;
    Tensor* results_tensor_;
    int num_results_;
};

}


#endif
