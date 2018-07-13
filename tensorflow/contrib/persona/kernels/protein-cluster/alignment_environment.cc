
#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_environment.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

  using namespace std;

  // len is seq1 len
  void AlignmentEnvironments::EstimPam(char* seq1, char* seq2, int len, double result[3]) const {

    EstimatePam(seq1, seq2, len, day_matrices_, (int)envs_.size(), logpam_env_.matrix, result);

  }
  
  const AlignmentEnvironment& AlignmentEnvironments::FindNearest(double pam) const {
    size_t i = 0;
    while (pam - envs_[i].pam_distance > 0.0f && i < envs_.size())
      i++;
    if (i == envs_.size())
      return envs_[i-1];
    else {
      if (fabs(envs_[i].pam_distance - pam) < fabs(envs_[i-1].pam_distance - pam))
        return envs_[i];
      else
        return envs_[i-1];
    }
  }
  
  const AlignmentEnvironment& AlignmentEnvironments::LogPamEnv() const {
    return logpam_env_;
  }

  const AlignmentEnvironment& AlignmentEnvironments::JustScoreEnv() const {
    return just_score_env_;
  }
  /*
  const SSW_Environment& AlignmentEnvironments::GetSSWEnv() const{
    return ssw_env_;
  }
  */
  
  void AlignmentEnvironments::Initialize(std::vector<AlignmentEnvironment>& envs, AlignmentEnvironment& logpam_env, 
      AlignmentEnvironment& just_score_env){//, SSW_Environment& ssw_env) {

    envs_ = envs;
    logpam_env_ = logpam_env;
    just_score_env_ = just_score_env;
    //ssw_env_ = ssw_env;
  }

  void AlignmentEnvironments::CreateDayMatrices(vector<double>& gap_open, vector<double>& gap_ext,
      vector<double>& pam_dist, vector<double*>& matrices) {
    day_matrices_ = createDayMatrices(&gap_open[0], &gap_ext[0], &pam_dist[0], (long long*)&matrices[0], (int)gap_open.size() - 1);
  }

}
