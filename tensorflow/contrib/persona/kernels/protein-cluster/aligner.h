#pragma once

#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_environment.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/params.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/swps3/extras.h"

namespace tensorflow {

class ProteinAligner {

  public:
    
    const static double pam_list_ [];

    struct Alignment {
      int seq1_min;
      int seq1_max;
      int seq2_min;
      int seq2_max;
      const AlignmentEnvironment* env;
      double score;
      double pam_distance;
      double pam_variance;
      int seq1_length; // len of aligned string minus '_'
      int seq2_length; // len of aligned string minus '_'
    };

    ProteinAligner(const AlignmentEnvironments* envs, const Parameters* params) : 
      envs_(envs), params_(params) {}

    ~ProteinAligner() { 
      if (buf1_) {
        delete [] buf1_;
      }
    }

    Status AlignLocal(const char* seq1, const char* seq2, int seq1_len, int seq2_len, Alignment& result);

    bool PassesThreshold(const char* seq1, const char* seq2, int seq1_len, int seq2_len);

    // with full range calc
    Status AlignDouble(const char* seq1, const char* seq2, int seq1_len, int seq2_len,
        bool stop_at_threshold,  Alignment& result, const AlignmentEnvironment& env);

    //const AlignmentEnvironments* Envs() { return envs_; }

    const Parameters* Params() { return params_; }

  private:
    const AlignmentEnvironments* envs_;
    const Parameters* params_;

    struct StartPoint {
      Alignment alignment;
      double estimated_pam;
      char* seq1;
      char* seq2;
      int seq1_len;
      int seq2_len;
    };
    
    void FindStartingPoint(const char*seq1, const char* seq2, int seq1_len, int seq2_len, 
        StartPoint& point);

    int AlignStrings(double* matrix, char *s1, int len1, char *s2, int len2, 
        double escore, char *o1, char *o2, double maxerr, double gap_open, double gap_ext, BTData* data);

    double c_align_double_global(double* matrix, const char *s1, int ls1,
        const char *s2, int ls2, double gap_open, double gap_ext, BTData* data);

    // these are for reuse over align double/local methods
    char* buf1_ = nullptr; // MAXSEQLEN
    char* buf2_ = nullptr; // MAXSEQLEN
    char* savebuf1_ = nullptr; // MAXSEQLEN
    char* savebuf2_ = nullptr; // MAXSEQLEN

    BTData bt_data_;

};

}
