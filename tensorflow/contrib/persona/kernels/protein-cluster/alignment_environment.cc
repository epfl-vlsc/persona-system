
#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_environment.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

  using namespace std;

  void AlignmentEnvironments::EstimatePam(char* seq1, char* seq2, int len) {

  }
  
  const AlignmentEnvironment& AlignmentEnvironments::FindNearest(double pam) const {

  }
  
  const AlignmentEnvironment& AlignmentEnvironments::LogPamEnv() const {
    return logpam_env_;
  }

  const AlignmentEnvironment& AlignmentEnvironments::JustScoreEnv() const {
    return just_score_env_;
  }
  
  void AlignmentEnvironments::Initialize(std::vector<AlignmentEnvironment>& envs, AlignmentEnvironment& logpam_env, 
      AlignmentEnvironment& just_score_env) {

    envs_ = envs;
    logpam_env_ = logpam_env;
    just_score_env_ = just_score_env;
  }

  void AlignmentEnvironments::CreateDayMatrices(vector<double>& gap_open, vector<double>& gap_ext,
      vector<double>& pam_dist, vector<double*>& matrices) {
    day_matrices_ = createDayMatrices(&gap_open[0], &gap_ext[0], &pam_dist[0], (long long*)&matrices[0], (int)gap_open.size() - 1);
  }

}
