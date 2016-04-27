
#ifndef TENSORFLOW_USER_OPS_SNAP_RESULTS_DECODE_H_
#define TENSORFLOW_USER_OPS_SNAP_RESULTS_DECODE_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

// results are held in a 3 dimensional tensor
// a results tensor holds a batch of results
// each batch entry is a set of results for one read, where index 0
// is special and holds whether the first result is primary, and the 
// number of results. 
// To the user however, results are still 0-indexed
class MutableSnapResultsDecode {

  public:
    MutableSnapResultsDecode(Tensor* results_tensor);

    // helper to get appropriate shape for given batch size and 
    // secondary results size
    static TensorShape get_results_shape(int batch_size,
        int num_secondary_results) {
      return TensorShape({batch_size, num_secondary_results+2, 5});
    }

    // set whether or not first result is primary
    void set_first_is_primary(int batch_index, bool value);

    bool first_is_primary(int batch_index);
    // set the number of results for the result in the given
    // batch index
    void set_num_results(int batch_index, int value);

    int64 num_results(int batch_index);

    int64 result_type(int batch_index, int result_index);

    void set_result_type(int batch_index, int result_index, int64 value);

    int64 genome_location(int batch_index, int result_index);

    void set_genome_location(int batch_index, int result_index, int64 value);

    int64 score(int batch_index, int result_index);

    void set_score(int batch_index, int result_index, int64 value);

    int64 mapq(int batch_index, int result_index);
    void set_mapq(int batch_index, int result_index, int64 value);

    int64 direction(int batch_index, int result_index);

    void set_direction(int batch_index, int result_index, int64 value);

    size_t size() { return num_results_; }

  private:
    static const int kFirstIsPrimary = 0;
    static const int kNumResults = 1;
    static const int kResultType = 0;
    static const int kGenomeLocation = 1;
    static const int kScore = 2;
    static const int kMapq = 3;
    static const int kDirection = 4;
    Tensor* results_tensor_;
    int num_results_;
};

// immutable version
class SnapResultsDecode {

  public:
    SnapResultsDecode(const Tensor* results_tensor);

    // helper to get appropriate shape for given batch size and 
    // secondary results size
    static TensorShape get_results_shape(int batch_size,
        int num_secondary_results) {
      return TensorShape({batch_size, num_secondary_results+2, 5});
    }

    bool first_is_primary(int batch_index);

    int64 num_results(int batch_index);

    int64 result_type(int batch_index, int result_index);

    int64 genome_location(int batch_index, int result_index);

    int64 score(int batch_index, int result_index);

    int64 mapq(int batch_index, int result_index);

    int64 direction(int batch_index, int result_index);

    size_t size() { return num_results_; }

  private:
    static const int kFirstIsPrimary = 0;
    static const int kNumResults = 1;
    static const int kResultType = 0;
    static const int kGenomeLocation = 1;
    static const int kScore = 2;
    static const int kMapq = 3;
    static const int kDirection = 4;
    const Tensor* results_tensor_;
    int num_results_;
};
}


#endif
