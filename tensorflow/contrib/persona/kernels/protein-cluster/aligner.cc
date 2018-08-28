
#include "tensorflow/contrib/persona/kernels/protein-cluster/aligner.h"
#include <cfloat>
#include <vector>
#include <fstream>

#include <math.h> // milad

extern "C" {
#include "tensorflow/contrib/persona/kernels/protein-cluster/swps3/DynProgr_sse_short.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/swps3/DynProgr_sse_double.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/swps3/extras.h"
  #include "tensorflow/contrib/persona/kernels/protein-cluster/ssw/kseq.h"
  #include "tensorflow/contrib/persona/kernels/protein-cluster/ssw/ssw.h"
}

namespace tensorflow {
   
using namespace std;

string PrintNormalizedProtein(const char* seq, size_t len) {
  vector<char> scratch;
  scratch.resize(len);
  memcpy(&scratch[0], seq, len);
  for (size_t i = 0; i < len; i++) {
    scratch[i] = scratch[i] + 'A';
  }
  return string(&scratch[0], len);
}

char* denormalize(const char* str, int len) {
  auto ret = new char[(len + 1) * sizeof(char)];
  int i;

  for (i = 0; i < len; ++i) {
    ret[i] = 'A' + str[i];
  }

  ret[len] = '\0';

  return ret;
}

void debug_alignment(char* name, void* profile, const char* s2, int ls2,
    Options* options) {
  char* dns2 = denormalize(s2, ls2);
  printf(
      "Aligning %s: profile = %lu, s2 = %s, len(s2) = %d, gapOpen = %f, gapExt = %f, threshold = %f\n",
      name, (unsigned long) profile, dns2, ls2, options->gapOpen,
      options->gapExt, options->threshold);
  free(dns2);
}

void debug_profile(char* name, void* pb, const char* q, int queryLen,
    void* matrix, int isShort) {
  char* query = denormalize(q, queryLen);
  int i, j;

  printf(
      "Successfully created %s profile at %lu from query = %s, len(query) = %d\n",
      name, (uint64) pb, query, queryLen);
  printf("The matrix used for the %s profile: \n", name);

  for (i = 0; i < 26; ++i) {
    printf("[");
    for (j = 0; j < 26; ++j) {
      double val;
      if (isShort) {
        val = (double) ((SMatrix) matrix)[i * 26 + j];
      } else {
        val = (double) ((BMatrix) matrix)[i * 26 + j];
      }

      printf("%+.2f, ", val);
    }
    printf("]\n");
  }

  free(query);
}

const double ProteinAligner::pam_list_ [] = {35, 49, 71, 98, 115, 133, 152, 174, 200, 229, 262, 300};

static int filenum = 0;

void ProteinAligner::FindStartingPoint(const char*seq1, const char* seq2, int seq1_len, int seq2_len, 
    StartPoint& point) {

  if (!buf1_) {
    // assuming that one big alloc is more efficient
    buf1_ = new char[MAXSEQLEN*4];
    buf2_ = buf1_ + MAXSEQLEN;
    savebuf1_ = buf2_ + MAXSEQLEN;
    savebuf2_ = savebuf1_ + MAXSEQLEN;
  }

  double starting_pam = -1e10;
  Status s;
  Alignment alignment;
  double estim_result[3];
  point.seq1 = savebuf1_;
  point.seq2 = savebuf2_;

  /*ofstream file;
  file.open(string("dump/fsp_alignment_") + to_string(filenum) + string(".csv"));
  filenum++;
  file << seq1_len << ", " << seq2_len << "\n";*/

  for (const double& pam : pam_list_) {
    const auto& new_env = envs_->FindNearest(pam);
    //LOG(INFO) << "FSP env gap_open is " << new_env.gap_open;
    s = AlignDouble(seq1, seq2, seq1_len, seq2_len, false, alignment, new_env);
    //LOG(INFO) << "FSP align double score is " << alignment.score;
  
    /*file << alignment.seq1_min << ", " << alignment.seq1_max << ", " << alignment.seq2_min 
      << ", " << alignment.seq2_max << "\n";*/

    char* s1 = const_cast<char*>(seq1) + alignment.seq1_min;
    char* s2 = const_cast<char*>(seq2) + alignment.seq2_min;
    int s1_len = alignment.seq1_max - alignment.seq1_min  + 1; //  ?
    int s2_len = alignment.seq2_max - alignment.seq2_min  + 1; //  ?
    //LOG(INFO) << "FSP: s1len: " << s1_len << " s2len: " << s2_len;

    int max_len = AlignStrings(new_env.matrix, s1, s1_len, s2, s2_len, alignment.score,
        buf1_, buf2_, 0.5e-4, new_env.gap_open, new_env.gap_extend, &bt_data_);
    envs_->EstimPam(buf1_, buf2_, max_len, estim_result);
    //LOG(INFO) << "FSP: estim pam values: " << estim_result[0] << ", " << estim_result[1] 
      //<< ", " << estim_result[2] << " str max len = " << max_len;

    if (estim_result[0] > starting_pam) {
      starting_pam = estim_result[0];
      memcpy(savebuf1_, buf1_, max_len);
      memcpy(savebuf2_, buf2_, max_len);
      point.estimated_pam = estim_result[0];
      point.alignment = alignment;
      point.alignment.pam_distance = estim_result[1];
      point.alignment.pam_variance = estim_result[2];
      point.seq1_len = max_len;
      point.seq2_len = max_len;
      //point.score = alignment.score;
      //point.env = &new_env;
      //point.pam_dist = estim_result[1];
      //point.pam_var = estim_result[2];
    }

  }
  //file.close();
}

int CountUnderscore(const char* buf, int len) {
  int sum = 0;
  for (size_t i = 0; i < len; i++) {
    if (buf[i] == '_') sum++;
  }
  return sum;
}

Status ProteinAligner::AlignSingle(const char*seq1, const char* seq2, int seq1_len, int seq2_len, 
    Alignment& result) {

  Status s;
  const auto& env = envs_->JustScoreEnv();
  s = AlignDouble(seq1, seq2, seq1_len, seq2_len, false, result, env);

  return s;
}

Status ProteinAligner::AlignLocal(const char*seq1, const char* seq2, int seq1_len, int seq2_len, 
    Alignment& result) {

  // find starting point
  // env find nearest point
  Status s;
  StartPoint point;
  Alignment alignment;
  double estim_result[3];
  FindStartingPoint(seq1, seq2, seq1_len, seq2_len, point);

  while (true) {
    const auto& new_env = envs_->FindNearest(point.alignment.pam_distance);
    //LOG(INFO) << "AlignLocal: env gap open is: " << new_env.gap_open;

    if (point.alignment.env == &new_env) // envs are basically static, so this is safe
      break;
   
    s = AlignDouble(seq1, seq2, seq1_len, seq2_len, false, alignment, new_env);
    //LOG(INFO) << "AlignLocal align double score is " << alignment.score;
    
    char* s1 = const_cast<char*>(seq1) + alignment.seq1_min;
    char* s2 = const_cast<char*>(seq2) + alignment.seq2_min;
    int s1_len = alignment.seq1_max - alignment.seq1_min + 1; //  ?
    int s2_len = alignment.seq2_max - alignment.seq2_min + 1; //  ?
    //LOG(INFO) << "AlignLocal: s1len: " << s1_len << " s2len: " << s2_len;

    int max_len = AlignStrings(new_env.matrix, s1, s1_len, s2, s2_len, alignment.score,
        buf1_, buf2_, 0.5e-4, new_env.gap_open, new_env.gap_extend, &bt_data_);
    envs_->EstimPam(buf1_, buf2_, max_len, estim_result);
    //LOG(INFO) << "AlignLocal: estim pam values: " << estim_result[0] << ", " << estim_result[1] 
     // << ", " << estim_result[2];

    point.seq1_len = max_len;
    point.seq2_len = max_len;
    memcpy(savebuf1_, buf1_, max_len);
    memcpy(savebuf2_, buf2_, max_len);
    //point.score = alignment.score;
    //point.env = &new_env;
    point.alignment = alignment;
    point.estimated_pam = estim_result[0];
    point.alignment.pam_distance = estim_result[1];
    point.alignment.pam_variance = estim_result[2];
    //point.pam_dist = estim_result[1];
    //point.pam_var = estim_result[2];

  }

  result = point.alignment;
  /*result.score = point.score;
  result.env = point.env;
  result.pam_distance = point.pam_dist;
  result.pam_variance = point.pam_var;
  result.seq1_max = alignment.seq1_max;
  result.seq2_max = alignment.seq2_max;
  result.seq1_min = alignment.seq1_min;
  result.seq2_min = alignment.seq2_min;*/

  result.seq1_length = point.seq1_len - CountUnderscore(point.seq1, point.seq1_len);
  result.seq2_length = point.seq2_len - CountUnderscore(point.seq2, point.seq2_len);
  
  return Status::OK();
}
    
Status ProteinAligner::AlignDouble(const char* seq1, const char* seq2, int seq1_len, int seq2_len, 
    bool stop_at_threshold, Alignment& result, const AlignmentEnvironment& env) {

  ProfileDouble* profile = createProfileDoubleSSE(seq1, seq1_len, env.matrix);

  int max1, max2;
  auto thresh = stop_at_threshold ? env.threshold : FLT_MAX;

  double score = align_double_local(profile, seq2, seq2_len, env.gap_open,
		env.gap_extend, thresh, &max1, &max2, &bt_data_);
  max1--;
  max2--;

  //LOG(INFO) << "aln dbl score " << score << ", max1, max2: " << max1 << ", " << max2;

  string seq1_rev(seq1, max1+1);
  reverse(seq1_rev.begin(), seq1_rev.end());
  string seq2_rev(seq2, max2+1);
  reverse(seq2_rev.begin(), seq2_rev.end());

  //LOG(INFO) << "rev 1 is " << PrintNormalizedProtein(seq1_rev.c_str(), seq1_rev.length());
  //LOG(INFO) << "rev 2 is " << PrintNormalizedProtein(seq2_rev.c_str(), seq2_rev.length());

  ProfileDouble* profile_rev = createProfileDoubleSSE(seq1_rev.c_str(), seq1_rev.length(), env.matrix);
  int max1_rev, max2_rev;

  double score_rev = align_double_local(profile_rev, seq2_rev.c_str(), seq2_rev.length(), env.gap_open,
		env.gap_extend, thresh, &max1_rev, &max2_rev, &bt_data_);
  max1_rev--;
  max2_rev--;
  
  //LOG(INFO) << "aln dbl rev score " << score_rev << ", max1, max2: " << max1_rev << ", " << max2_rev;

  result.score = score;
  result.env = &env;
  result.seq1_max = max1;
  result.seq2_max = max2;
  result.seq1_min = max1 - max1_rev;
  result.seq2_min = max2 - max2_rev;
  //LOG(INFO) << "seq1max: " << max1 << ", seq2max: " << max2 << ", seq1min: " << result.seq1_min 
    //<< ", seq2min: " << result.seq2_min;

  CHECK_LE(fabs(score - score_rev), fabs(score)*1e-10);

  free_profile_double_sse(profile);
  free_profile_double_sse(profile_rev);

  return Status::OK();

}
    
bool ProteinAligner::PassesThreshold(const char* seq1, const char* seq2, int seq1_len, int seq2_len) {

  /*
  cout << " Seq 1 ";  
  for (int r = 0; r<seq1_len; r++){
    cout << seq1_denorm[r] ;
    }
    cout<< endl; 
   */ 
  // we use the short (int16) version for this
  const auto& env = envs_->JustScoreEnv();
  ProfileShort* profile = swps3_createProfileShortSSE(seq1, seq1_len, env.matrix_int16);

  Options options;
  options.gapOpen = env.gap_open_int16;
  options.gapExt= env.gap_ext_int16;
  options.threshold= env.threshold;
  
  //debug_profile("SHORT", profile, seq1, seq1_len, env.matrix_int16, 1);
  //debug_alignment("SHORT", profile, seq2, seq2_len, &options);

  double score = swps3_alignmentShortSSE(profile, seq2, seq2_len, &options);
  double value;
  if (score >= FLT_MAX) 
    value = SHRT_MAX;
  else
    value = score / (65535.0f / options.threshold);
  // cout << value<<", " ;

  swps3_freeProfileShortSSE(profile);
  //LOG(INFO) << "value is " << value << " score is " << score;
  return value >= 0.75f * params_->min_score;
}

bool isPowerof2(unsigned int x) {
    return x && !(x & (x - 1));
}

//bool ProteinAligner::PassesThresholdSSW(const char* seq1_norm, const char* seq2_norm, int seq1_len, int seq2_len) {

//typedef struct {
//    bool passed;
//    int32_t ref_begin1;
//    int32_t ref_end1;
//    int32_t	read_begin1;
//    int32_t read_end1;
//} m_threshold; // milad_threshold

// bool ProteinAligner::PassesThresholdSSW(const char* seq1_norm, const char* seq2_norm, int seq1_len, int seq2_len) {
m_threshold ProteinAligner::PassesThresholdSSW(const char* seq1_norm, const char* seq2_norm, int seq1_len, int seq2_len) {

  //Constants Being Initialised
  
  int32_t l, m, k, match = 2, mismatch_ssw = 2, gap_open = 37.64 - 7.434 * log10(224), gap_extension = 1.3961, n = 5, s1 = 1024, s2 = 1024, filter = 0;
  int8_t* mata = (int8_t*)calloc(25, sizeof(int8_t));
  // int8_t mat = (int8_t*)calloc(25, sizeof(int8_t));  
  const int8_t* mat = mata;
  int8_t* ref_num = (int8_t*)malloc(s1);
  int8_t* num = (int8_t*)malloc(s2);

//    int8_t aa_table[128] = {
//            23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
//            23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
//            23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
//            23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
//            23, 0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23,
//            14, 5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23,
//            23, 0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23,
//            14, 5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23
//    };

  //Table for Protein Matchings
  static const int8_t aa_table[128] = { // milad modified table for normalized input (optimization)
    0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23, 14,
    5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23, 23,
    0,  20, 4,  3,  6,  13, 7,  8,  9,  23, 11, 10, 12, 2,  23, 14,
    5,  1,  15, 16, 23, 19, 17, 22, 18, 21, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
    23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
  };

  // cout << 335 <<endl;
  //Alignment Matrix
  static const int8_t mat50[] = {
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

  
  for (l = k = 0; LIKELY(l < 4); ++l) {
      for (m = 0; LIKELY(m < 4); ++m) mata[k++] = l == m ? match : -mismatch_ssw; // weight_match : -weight_mismatch_ssw 
      mata[k++] = 0; // ambiguous base
  }
  // cout << 369 <<endl;
  for (m = 0; LIKELY(m < 5); ++m) mata[k++] = 0; //387,385,384 mata -> mat
  n = 24;
//  int8_t* table = aa_table;
  const int8_t* table = aa_table; // milad
  // table = aa_table;
  mat = mat50;


  //To be done every time // milad not anymore (optimization)
  //const char * seq1 = denormalize(seq1_norm, seq1_len);
  //const char * seq2 = denormalize(seq2_norm, seq2_len);
  const char * seq1 = seq1_norm; // watch out for \0 at the end of denormalized form
  const char * seq2 = seq2_norm;

  //SSW_Environment ssw_env_pair = envs_ ->GetSSWEnv();
  s_profile* p= 0;
  int32_t readLen = (int32_t)seq2_len;
  int32_t maskLen = readLen / 2;
  // cout << match <<endl;
  // cout << 378 <<endl;

    // milad
//  while (readLen >= s2) {
//      // cout << "s2: "<<s2<<endl;
//      ++s2;
//      kroundup32(s2);
//      num = (int8_t*)realloc(num, s2);
//
//  }
    // milad
    if (readLen >= s2) {
        int32_t temp = readLen;
        if (isPowerof2(readLen))
            temp ++; // force it to roundup to next power of two
        kroundup32(temp);
        num = (int8_t*)realloc(num, temp);
    }

  // cout <<393<<endl;
  for (int m = 0; m < readLen; ++m) num[m] = table[(int)seq2[m]];
    // cout <<394<<endl;
  p = ssw_init(num, readLen, mat, n, 2);
  // cout << 387 <<endl;
  s_align* result = 0;
  int32_t refLen = (int32_t)seq1_len;
  int8_t flag = 0;
  // cout <<398<<endl;
  // milad
//  while (refLen > s1) {
//      // cout << "s1: "<<s1<<endl;
//      ++s1;
//      kroundup32(s1);
//      ref_num = (int8_t*)realloc(ref_num, s1);
//  }
    //milad
    if (refLen >= s1) {
        int32_t temp = refLen;
        kroundup32(temp);
        ref_num = (int8_t*)realloc(ref_num, temp);
    }

  for (int m = 0; m < refLen; ++m) ref_num[m] = table[(int)seq1[m]];
    // cout <<404<<endl;
    /*Printerere
  cout << "p: "<< p << endl;
  cout << "ref_num: "<< ref_num << endl;
  cout << "refLen: "<< refLen << endl;
  cout << "gap_open: "<< gap_open << endl;
  cout << "gap_extension: "<< gap_extension << endl;
  cout << "flag: "<< flag << endl;
  cout << "filter: "<< filter << endl;
  cout << "maskLen: "<< maskLen << endl;
  */

  result = ssw_align (p, ref_num, refLen, gap_open, gap_extension, flag, filter, 0, maskLen);
  init_destroy(p);
  // cout << result->score1<< endl;
  //ThresholdSSw

  //To be done once from here
  
  free(mata);
  free(ref_num);
  free(num);

  //To here

  bool retval;
  if (result-> score1 > 110){
    retval =  true;
  }
  else retval = false;

  m_threshold threshold; // milad TODO check performance. it's not a pointer!
  threshold.passed = retval;
  threshold.ref_begin1 = result->ref_begin1;
  threshold.ref_end1 = result->ref_end1;
  threshold.read_begin1 = result->read_begin1;
  threshold.read_end1 = result->read_end1; // TODO check segfault?

  align_destroy(result);
//  return retval;
  return threshold;
}

double ProteinAligner::c_align_double_global(double* matrix, const char *s1, int ls1,
    const char *s2, int ls2, double gap_open, double gap_ext, BTData* data) {
  int i, j, k;
  int AToInts2[MAXSEQLEN + 1];

  //double coldel[MAXSEQLEN+1], S[MAXSEQLEN+1];
  //int DelFrom[MAXSEQLEN+1];

  double DelFixed, DelIncr, *Score_s1 = NULL;
  double t, t2/*, MaxScore*/, rowdel, Sj1;
  /*double vScore[MAXMUTDIM];*/
  int NoSelf = 0;
  /* This totcells was a system variable and I have no idea what it is used for */
  double totcells = 0.0;

  DelFixed = gap_open;
  DelIncr = gap_ext;

  /*MaxScore = MINUSINF;*/
  data->S[0] = data->coldel[0] = 0;
  for (j = 1; j <= ls2; j++) {
    /*if (s2[j - 1] == '_')
     *     userror("underscores cannot be used in sequence alignment");*/
    AToInts2[j] = /*MapSymbol(s2[j - 1], DM)*/s2[j - 1];
    data->coldel[j] = MINUSINF;
    data->S[j] = /*(mode == CFE || mode == Local) ? 0 : */data->S[j - 1]
      + (j == 1 ? DelFixed : DelIncr);
  }

  data->DelFrom[0] = 1;
  for (i = 1; i <= ls1; i++) {

    Sj1 = data->S[0];
    data->coldel[0] += i == 1 ? DelFixed : DelIncr;
    data->S[0] = /*(mode == CFE || mode == Local) ? 0 :*/data->coldel[0];
    rowdel = MINUSINF;

    /*if (ProbSeqCnt == 0) {*/
    /* setup Score_s1 */
    /*if (s1[i - 1] == '_')
     *     userror("underscores cannot be used in sequence alignment");*/
    k = /*MapSymbol(s1[i - 1], DM)*/s1[i - 1];
    Score_s1 = matrix + k * MATRIX_DIM;
    /*} else if (ProbSeqCnt == 1) {
     *     ProbScore(s1.ps + (i - 1) * NrDims, M, AF, NrDims, vScore);
     *         Score_s1 = vScore;
     *             }*/

    /* Complete code for the inner loop of dynamic programming.
     *     Any changes should be made to this code and generate the
     *         others by explicitly eliminating the tests tested on the
     *             outside.  The purpose is to maximize speed of the most
     *                 common case.   March 27, 2005 */
    for (j = 1; j <= ls2; j++) {
      /* current row is i (1..ls1), column is j (1..ls2) */

      data->coldel[j] += DelIncr;
      t = (t2 = data->S[j]) + DelFixed;
      if (data->coldel[j] < t) {
        data->coldel[j] = t;
        data->DelFrom[j] = i;
      }

      rowdel += DelIncr;
      t = data->S[j - 1] + DelFixed;
      if (rowdel < t)
        rowdel = t;

      /* TODO: check that det vs prob can never be a Self */
      t = NoSelf /*&& ProbSeqCnt == 0*/&& s1 + i == s2 + j ?
        MINUSINF : Sj1 + Score_s1[AToInts2[j]];
      if (t < rowdel)
        t = rowdel;
      if (t < data->coldel[j])
        t = data->coldel[j];

      /*if (mode == Local) {
       *       if (t < 0)
       *             t = 0;
       *                   if (t > MaxScore) {
       *                         MaxScore = t;
       *                               *Max1 = i;
       *                                     *Max2 = j;
       *                                           if (MaxScore >= goal) {
       *                                                 totcells += j;
       *                                                       return (MaxScore);
       *                                                             }
       *                                                                   }
       *                                                                         } else if (mode == Shake && t > MaxScore) {
       *                                                                               MaxScore = t;
       *                                                                                     *Max1 = i;
       *                                                                                           *Max2 = j;
       *                                                                                                 if (MaxScore >= goal) {
       *                                                                                                       totcells += j;
       *                                                                                                             return (MaxScore);
       *                                                                                                                   }
       *                                                                                                                         }*/
      data->S[j] = t;
      Sj1 = t2;
    }
    totcells += ls2;

    /*if ((mode == CFE || mode == CFEright) && S[ls2] > MaxScore) {
     *     MaxScore = S[ls2];
     *         *Max1 = i;
     *             *Max2 = ls2;
     *                 if (MaxScore >= goal)
     *                     return (MaxScore);
     *                         }*/

  }

  /*if (mode == CFE || mode == CFEright)
   *   for (j = 1; j <= ls2; j++)
   *     if (S[j] > MaxScore) {
   *       MaxScore = S[j];
   *         *Max1 = ls1;
   *           *Max2 = j;
   *             }
   *
   *               if (mode == Global) {
   *                 *Max1 = ls1;
   *                   *Max2 = ls2;*/
  return (data->S[ls2]);
  /*}
   *   return (MaxScore);*/
}
// copied directly from pyopa python extension
int ProteinAligner::AlignStrings(double* matrix, char *s1, int len1, char *s2, int len2, 
        double escore, char *o1, char *o2, double maxerr, double gap_open, double gap_ext, BTData* data) {
  int DelFrom1[MAXSEQLEN + 1], i, i1, j/*, M1*/;
  double S1[MAXSEQLEN + 1], coldel1[MAXSEQLEN + 1];
  double Slen2i, maxs, t;
  char rs1[2 * MAXSEQLEN], rs2[MAXSEQLEN], *prs1, *prs2;
  double tot;
  
  //double coldel[MAXSEQLEN+1], S[MAXSEQLEN+1];
  //int DelFrom[MAXSEQLEN+1];
  /*int Global = 2;*/
  /* this NoSelf could be a parameter */
  /*int NoSelf = 0;*/

  /* make len1 >= len2 */
  if (len1 < len2) {
    prs1 = s1;
    s1 = s2;
    s2 = prs1;
    i = len1;
    len1 = len2;
    len2 = i;
    prs1 = o1;
    o1 = o2;
    o2 = prs1;
  }
  /*ASSERT( len2 >= 0, Newint(len2) );*/

  /* s2 is null */
  if (len2 == 0) {
    /*if( escore != DeletionCost(len1, gap_open, gap_ext) )
     *     userror("score incompatible with aligned strings");*/
    for (i = 0; i < len1; i++) {
      o1[i] = s1[i];
      o2[i] = '_';
    }
    return (len1);
  }

  /* 1 against 1 (this case is needed for recursion termination) */
  if (len1 == 1) {
    if (2 * gap_open >= escore - maxerr) {
      o1[0] = s1[0];
      o2[0] = '_';
      o1[1] = '_';
      o2[1] = s2[0];
      return (2);
    }
    /*if( (NoSelf && s1==s2) || escore > DMScore(s1[0],s2[0],matrix) )
     *     userror("Alignment/Match has incorrect data");*/
    o1[0] = s1[0];
    o2[0] = s2[0];
    return (1);
  }

  /* equal length, try to see if there is a match without deletions */
  if (len1 == len2 && s1 != s2) {
    for (i = tot = 0; i < len1; i++)
      tot += /*DMScore(s1[i],s2[i],matrix)*/matrix[s1[i] * MATRIX_DIM
        + s2[i]];
    if (tot >= escore) {
      for (i = 0; i < len1; i++) {
        o1[i] = s1[i];
        o2[i] = s2[i];
      }
      return (len1);
    }
  }

  /* len1 >= len2 >= 1, try to see if there is an all deleted match */
  /* allow some tolerance for a Match read with score error */
  if (2 * gap_open + (len1 + len2 - 2) * gap_ext >= escore - maxerr) {
    for (i = 0; i < len1; i++) {
      o1[i] = s1[i];
      o2[i] = '_';
    }
    for (i = 0; i < len2; i++) {
      o1[len1 + i] = '_';
      o2[len1 + i] = s2[i];
    }
    return (len1 + len2);
  }

  /* reverse strings */
  if (s1 - s2 >= 0 && s2 + len2 - s1 > 0) {
    j = MMAX(s1 - s2 + len1, len2);
    for (i = 0; i < j; i++)
      rs1[j - 1 - i] = s2[i];
    prs1 = rs1 + (j + s2 - s1 - len1);
    prs2 = rs1 + (j - len2);
  } else if (s2 - s1 >= 0 && s1 + len1 - s2 > 0) {
    j = MMAX(s2 - s1 + len2, len1);
    for (i = 0; i < j; i++)
      rs1[j - 1 - i] = s1[i];
    prs2 = rs1 + (j + s1 - s2 - len2);
    prs1 = rs1 + (j - len1);
  } else {
    for (i = 0; i < len2; i++)
      rs2[len2 - 1 - i] = s2[i];
    for (i = 0; i < len1; i++)
      rs1[len1 - 1 - i] = s1[i];
    prs1 = rs1;
    prs2 = rs2;
  }

  /* divides s1 in half */
  i1 = len1 / 2;
  /*seq1.ds = s1;
   *   seq2.ds = s2;*/
  c_align_double_global(matrix, s1, i1, s2, len2, gap_open, gap_ext, data);
  for (i = 0; i <= len2; i++) {
    S1[i] = data->S[i];
    coldel1[i] = data->coldel[i];
    DelFrom1[i] = data->DelFrom[i];
  }
  /*seq1.ds = prs1;
   *   seq2.ds = prs2;*/
  c_align_double_global(matrix, prs1, len1 - i1, prs2, len2, gap_open,
      gap_ext, data);

  /* Find the best sum of scores from the two halves */
  maxs = -DBL_MAX;
  for (j = 0; j <= len2; j++) {
    t = S1[j] + data->S[len2 - j];
    if (t > maxs) {
      maxs = t;
      i = j;
    }
    t = coldel1[j] + data->coldel[len2 - j] - gap_open + gap_ext;
    if (t > maxs) {
      maxs = t;
      i = j;
    }
  }
  /*if( !(maxerr != 0 || maxs==escore) ) {
   *   printf( "DM->bits=%d, gap_open=%.18g, gap_ext=%.18g\n",
   *   DM->bits, gap_open, gap_ext );
   *       printf( "maxerr=%.18g, maxs=%.18g, escore=%.18g\n",
   *         maxerr, maxs, escore );
   *           }
   *             ASSERT2( maxerr != 0 || maxs==escore, NewNumber(maxs), NewNumber(escore) );
   *               if( ENVprintlevel > 0 && maxs < escore-maxerr )
   *                 fprintf( w_unit, "Warning: DynProgStrings could not reach "
   *                   "score %.12g, reached %.12g instead\n", escore, maxs );*/

  /* splitting on a match */
  if (maxs == S1[i] + data->S[len2 - i]) {
    Slen2i = data->S[len2 - i];
    j = AlignStrings(matrix, s1, i1, s2, i, S1[i], o1, o2, 0.0, gap_open,
        gap_ext, data);
    j += AlignStrings(matrix, s1 + i1, len1 - i1, s2 + i, len2 - i,
        Slen2i, o1 + j, o2 + j, 0.0, gap_open, gap_ext, data);
    return (j);
  }

  /* splitting on a vertical deletion */
  {
    int i3, i4, len;
    i3 = DelFrom1[i] - 1;
    i4 = len1 - data->DelFrom[len2 - i] + 2;
    Slen2i = data->coldel[len2 - i];
    len = AlignStrings(matrix, s1, i3, s2, i,
        coldel1[i] - gap_open - gap_ext * (i1 - i3 - 1), o1, o2, 0.0,
        gap_open, gap_ext, data);
    for (j = i3 + 1; j < i4; j++) {
      o1[len] = s1[j - 1];
      o2[len++] = '_';
    }
    len += AlignStrings(matrix, s1 + i4 - 1, len1 - i4 + 1, s2 + i,
        len2 - i, Slen2i - gap_open - gap_ext * (i4 - i1 - 2), o1 + len,
        o2 + len, 0.0, gap_open, gap_ext, data);
    return (len);
  }
  return 0;
}

}
