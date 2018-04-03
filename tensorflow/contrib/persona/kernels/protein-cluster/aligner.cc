
#include "tensorflow/contrib/persona/kernels/protein-cluster/aligner.h"
#include <cfloat>
extern "C" {
#include "tensorflow/contrib/persona/kernels/protein-cluster/swps3/DynProgr_sse_short.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/swps3/DynProgr_sse_double.h"
}

namespace tensorflow {

Status ProteinAligner::AlignLocal(const char*seq1, const char* seq2, int seq1_len, int seq2_len, 
    Alignment& result) {
  
}
    
Status ProteinAligner::AlignDouble(const char* seq1, const char* seq2, int seq1_len, int seq2_len, 
    bool stop_at_threshold, Alignment& result, const AlignmentEnvironment& env) {

  ProfileDouble* profile = createProfileDoubleSSE(seq1, seq1_len, env.matrix);

  int max1, max2;
  auto thresh = stop_at_threshold ? env.threshold : FLT_MAX;

  double score = align_double_local(profile, seq2, seq2_len, env.gap_open,
		env.gap_extend, thresh, &max1, &max2);
  max1--;
  max2--;

  string seq1_rev(seq1, max1);
  reverse(seq1_rev.begin(), seq1_rev.end());
  string seq2_rev(seq2, max2);
  reverse(seq2_rev.begin(), seq2_rev.end());

  ProfileDouble* profile_rev = createProfileDoubleSSE(seq1_rev.c_str(), max1, env.matrix);
  int max1_rev, max2_rev;

  double score_rev = align_double_local(profile_rev, seq2_rev.c_str(), max2, env.gap_open,
		env.gap_extend, thresh, &max1_rev, &max2_rev);
  max1_rev--;
  max2_rev--;

  CHECK_LE(fabs(score - score_rev), fabs(score)*1e-10);

  result.score = score;
  result.seq1_max = max1;
  result.seq2_max = max2;
  result.seq1_min = max1 - max1_rev;
  result.seq2_min = max2 - max2_rev;

  free_profile_double_sse(profile);
  free_profile_double_sse(profile_rev);

  return Status::OK();

}
    
bool ProteinAligner::PassesThreshold(const char* seq1, const char* seq2, int seq1_len, int seq2_len) {

  const auto& env = envs_->JustScoreEnv();
  ProfileShort* profile = swps3_createProfileShortSSE(seq1, seq1_len, env.matrix_int16);

  Options options;
  options.gapOpen = env.gap_open;
  options.gapExt= env.gap_extend;
  options.threshold= env.threshold;

  double score = swps3_alignmentShortSSE(profile, seq2, seq2_len, &options);
  short value;
  if (score >= FLT_MAX) 
    value = SHRT_MAX;
  else
    value = short(score / (65535.0f / options.threshold));

  swps3_freeProfileShortSSE(profile);
  return value >= 0.75f * params_->min_score;
}

}
