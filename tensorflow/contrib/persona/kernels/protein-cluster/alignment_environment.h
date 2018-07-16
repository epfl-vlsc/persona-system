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

struct SSW_Environment{
  int32_t l, m, k, match, mismatch_ssw, gap_open, gap_extension, n, s1, s2, filter;
  int8_t* mata ; 
  const int8_t* mat;
  int8_t* ref_num;
  int8_t* num;
  //Table for Protein Matchings
  int8_t aa_table[128] = {
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23,
    14, 5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23,
    23, 0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23,
    14, 5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23
  };
  //Alignment Matrix
  //static const int8_t mat50[] = {}; //static const
  //Alignment Matrix
  int8_t mat50[576] = {
     //PAM220
      2,  -2, 0,  0,  -2, -1, 0,  1,  -2, -1, -2, -1, -1, -4, 1,  1,  1,  -6, -4, 0,  0,  0,  0,  -8,
      -2, 7,  0,  -2, -4, 1,  -1, -3, 2,  -2, -3, 4,  -1, -5, 0,  0,  -1, 2,  -5, -3, -1, 0,  -1, -8,
      0,  0,  3,  2,  -4, 1,  2,  0,  2,  -2, -3, 1,  -2, -4, -1, 1,  0,  -4, -2, -2, 2,  1,  0,  -8,
      0,  -2, 2,  4,  -6, 2,  4,  0,  1,  -3, -5, 0,  -3, -6, -1, 0,  0,  -8, -5, -3, 4,  3,  -1, -8,
      -2, -4, -4, -6, 12, -6, -6, -4, -4, -3, -7, -6, -6, -5, -3, 0,  -3, -8, 0,  -2, -5, -6, -3, -8,
      -1, 1,  1,  2,  -6, 5,  3,  -2, 3,  -2, -2, 1,  -1, -5, 0,  -1, -1, -5, -5, -2, 1,  4,  -1, -8,
      0,  -1, 2,  4,  -6, 3,  4,  0,  1,  -2, -4, 0,  -2, -6, -1, 0,  -1, -8, -5, -2, 3,  4,  -1, -8,
      1,  -3, 0,  0,  -4, -2, 0,  5,  -3, -3, -5, -2, -3, -5, -1, 1,  0,  -8, -6, -2, 0,  -1, -1, -8,
      -2, 2,  2,  1,  -4, 3,  1,  -3, 7,  -3, -2, 0,  -3, -2, 0,  -1, -2, -3, 0,  -3, 1,  2,  -1, -8,
      -1, -2, -2, -3, -3, -2, -2, -3, -3, 5,  2,  -2, 2,  1,  -2, -2, 0,  -6, -1, 4,  -2, -2, -1, -8,
      -2, -3, -3, -5, -7, -2, -4, -5, -2, 2,  6,  -3, 4,  2,  -3, -3, -2, -2, -1, 2,  -4, -3, -2, -8,
      -1, 4,  1,  0,  -6, 1,  0,  -2, 0,  -2, -3, 5,  1,  -6, -1, 0,  0,  -4, -5, -3, 0,  0,  -1, -8,
      -1, -1, -2, -3, -6, -1, -2, -3, -3, 2,  4,  1,  8,  0,  -2, -2, -1, -5, -3, 2,  -3, -2, -1, -8,
      -4, -5, -4, -6, -5, -5, -6, -5, -2, 1,  2,  -6, 0,  10, -5, -4, -4, 0,  7,  -2, -5, -6, -3, -8,
      1,  0,  -1, -1, -3, 0,  -1, -1, 0,  -2, -3, -1, -2, -5, 7,  1,  0,  -6, -6, -1, -1, 0,  -1, -8,
      1,  0,  1,  0,  0,  -1, 0,  1,  -1, -2, -3, 0,  -2, -4, 1,  2,  2,  -3, -3, -1, 0,  0,  0,  -8,
      1,  -1, 0,  0,  -3, -1, -1, 0,  -2, 0,  -2, 0,  -1, -4, 0,  2,  3,  -6, -3, 0,  0,  -1, 0,  -8,
      -6, 2,  -4, -8, -8, -5, -8, -8, -3, -6, -2, -4, -5, 0,  -6, -3, -6, 17, 0,  -7, -6, -7, -5, -8,
      -4, -5, -2, -5, 0,  -5, -5, -6, 0,  -1, -1, -5, -3, 7,  -6, -3, -3, 0,  11, -3, -3, -5, -3, -8,
      0,  -3, -2, -3, -2, -2, -2, -2, -3, 4,  2,  -3, 2,  -2, -1, -1, 0,  -7, -3, 5,  -2, -2, -1, -8,
      0,  -1, 2,  4,  -5, 1,  3,  0,  1,  -2, -4, 0,  -3, -5, -1, 0,  0,  -6, -3, -2, 3,  2,  -1, -8,
      0,  0,  1,  3,  -6, 4,  4,  -1, 2,  -2, -3, 0,  -2, -6, 0,  0,  -1, -7, -5, -2, 2,  4,  -1, -8,
      0,  -1, 0,  -1, -3, -1, -1, -1, -1, -1, -2, -1, -1, -3, -1, 0,  0,  -5, -3, -1, -1, -1, -1, -8,
      -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, 1
    };
  int8_t* table;
};

class AlignmentEnvironments {
 // pointers here own no data
 public:
  AlignmentEnvironments() {}
  void EstimPam(char* seq1, char* seq2, int len, double result[3]) const;
  const AlignmentEnvironment& FindNearest(double pam) const;
  const AlignmentEnvironment& LogPamEnv() const;
  const AlignmentEnvironment& JustScoreEnv() const;
  const SSW_Environment& GetSSWEnv() const;

  // init methods
  void CreateDayMatrices(std::vector<double>& gap_open, std::vector<double>& gap_ext,
      std::vector<double>& pam_dist, std::vector<double*>& matrices);
  void Initialize(std::vector<AlignmentEnvironment>& envs, AlignmentEnvironment& logpam_env, 
      AlignmentEnvironment& just_score_env, SSW_Environment& ssw_env);

 private:
  std::vector<AlignmentEnvironment> envs_;
  DayMatrix* day_matrices_;
  //double* logpam1_matrix_;
  AlignmentEnvironment logpam_env_;
  AlignmentEnvironment just_score_env_;
  SSW_Environment ssw_env_;

};

}

