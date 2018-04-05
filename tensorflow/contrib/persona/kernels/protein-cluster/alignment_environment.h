#pragma once
#include "tensorflow/core/lib/core/errors.h"
extern "C" {
#include "swps3/EstimatePam.h"
}
#include <vector>

namespace tensorflow {

struct AlignmentEnvironment {
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
 // pointers here own no data
 public:
  AlignmentEnvironments() {}
  void EstimPam(char* seq1, char* seq2, int len, double result[3]);
  const AlignmentEnvironment& FindNearest(double pam) const;
  const AlignmentEnvironment& LogPamEnv() const;
  const AlignmentEnvironment& JustScoreEnv() const;

  // init methods
  void CreateDayMatrices(std::vector<double>& gap_open, std::vector<double>& gap_ext,
      std::vector<double>& pam_dist, std::vector<double*>& matrices);
  void Initialize(std::vector<AlignmentEnvironment>& envs, AlignmentEnvironment& logpam_env, 
      AlignmentEnvironment& just_score_env);

 private:
  std::vector<AlignmentEnvironment> envs_;
  DayMatrix* day_matrices_;
  //double* logpam1_matrix_;
  AlignmentEnvironment logpam_env_;
  AlignmentEnvironment just_score_env_;
};

}

