#pragma once
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/contrib/persona/kernels/protein_clustering/swps3/EstimatePam.h"
#include <vector>

namespace tensorflow {

class AlignmentEnvironment {
    double gap_open;
    double gap_extend;
    double pam_distance;
    double threshold;
    double* matrix = nullptr;  // 26 by 26 matrix of scores
    int8 gap_open_int8;
    int8 gap_ext_int8;
    int8* matrix_int8 = nullptr;
    int16 gap_open_int16;
    int16 gap_ext_int16;
    int16* matrix_int16 = nullptr;
};

class AlignmentEnvironments {
 public:
  void EstimatePam(char* seq1, char* seq2, int len);
  const AlignmentEnvironment& FindNearest(double pam);

 private:
  std::vector<AlignmentEnvironment> envs_;
  DayMatrix* day_matrices_;
  double* logpam1_matrix_;
}

}

