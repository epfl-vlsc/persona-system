#pragma once

#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_environment.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/params.h"

namespace tensorflow {

class ProteinAligner {

  public:

    struct Alignment {
      int seq1_min;
      int seq1_max;
      int seq2_min;
      int seq2_max;
      const AlignmentEnvironment* env;
      int score;
      double pam_distance;
      double pam_variance;
    };

    ProteinAligner(const AlignmentEnvironments* envs, const Parameters* params) : 
      envs_(envs), params_(params) {}

    Status AlignLocal(const char* seq1, const char* seq2, int seq1_len, int seq2_len, Alignment& result);

    bool PassesThreshold(const char* seq1, const char* seq2, int seq1_len, int seq2_len);

    // with full range calc
    Status AlignDouble(const char* seq1, const char* seq2, int seq1_len, int seq2_len,
        bool stop_at_threshold,  Alignment& result, const AlignmentEnvironment& env);

    //const AlignmentEnvironments* Envs() { return envs_; }

  private:
    const AlignmentEnvironments* envs_;
    const Parameters* params_;
    static const pam_list = [35, 49, 71, 98, 115, 133, 152, 174, 200, 229, 262, 300];

};

}
