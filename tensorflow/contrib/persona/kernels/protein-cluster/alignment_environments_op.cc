
// Stuart Byma
// Op providing Alignment Environments
// is generally messy because the primary concern is 
// all the data being the same as the orig implementation

#include <algorithm>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/protein-cluster/alignment_environment.h"

namespace tensorflow {
using namespace std;
using namespace errors;

#define DIMSIZE 26
#define MDIM DIMSIZE*DIMSIZE

class AlignmentEnvironmentsOp : public OpKernel {
 public:
  typedef BasicContainer<AlignmentEnvironments> EnvsContainer;

  AlignmentEnvironmentsOp(OpKernelConstruction* context)
      : OpKernel(context), object_handle_set_(false) {

    vector<Tensor> double_matrices, int8_matrices, int16_matrices;
    Tensor logpam_matrix, gaps, gap_exts, pam_dists, thresholds, logpam_gap, 
           logpam_gap_ext, logpam_threshold, logpam_pam_dist;
    OP_REQUIRES_OK(context, context->GetAttr("double_matrices", &double_matrices));
    OP_REQUIRES_OK(context, context->GetAttr("gaps", &gaps));
    OP_REQUIRES_OK(context, context->GetAttr("gap_extends", &gap_exts));
    OP_REQUIRES_OK(context, context->GetAttr("pam_dists", &pam_dists));
    OP_REQUIRES_OK(context, context->GetAttr("thresholds", &thresholds));
    
    OP_REQUIRES_OK(context, context->GetAttr("logpam_gap", &logpam_gap));
    OP_REQUIRES_OK(context, context->GetAttr("logpam_gap_ext", &logpam_gap_ext));
    OP_REQUIRES_OK(context, context->GetAttr("logpam_pam_dist", &logpam_pam_dist));
    OP_REQUIRES_OK(context, context->GetAttr("logpam_threshold", &logpam_threshold));
    OP_REQUIRES_OK(context, context->GetAttr("logpam_matrix", &logpam_matrix));

    logpam_matrix_ = new double[MDIM];
    auto lpm = logpam_matrix.matrix<double>();
    for (size_t rows = 0; rows < logpam_matrix.dim_size(0); rows++) {
        for (size_t cols = 0; cols < logpam_matrix.dim_size(1); cols++) {
              logpam_matrix_[rows*DIMSIZE + cols] = lpm(rows, cols);
        }
    }
    logpam_gap_ = logpam_gap.scalar<double>()();
    logpam_gap_ext_ = logpam_gap_ext.scalar<double>()();
    logpam_threshold_ = logpam_gap_ext.scalar<double>()();
    logpam_pam_dist_ = logpam_pam_dist.scalar<double>()();
    logpam_int8_matrix_ = CreateScaled(logpam_matrix_, logpam_threshold_, logpam_gap_,
        logpam_gap_ext_, logpam_int8_gap_, logpam_int8_gap_ext_);
    logpam_int16_matrix_ = CreateScaled(logpam_matrix_, logpam_threshold_, logpam_gap_,
        logpam_gap_ext_, logpam_int16_gap_, logpam_int16_gap_ext_);
   
    auto sz = double_matrices.size();
    LOG(INFO) << "size is " << sz;
    // we can only pass floats and not doubles as attributes what the fuck 
    gaps_.reserve(sz);
    gap_extends_.reserve(sz); 
    pam_dists_.reserve(sz);
    thresholds_.reserve(sz);

    double_matrices_.resize(sz);
    int8_matrices_.resize(sz);
    int16_matrices_.resize(sz);
    int8_gaps_.resize(sz);
    int16_gaps_.resize(sz);
    int8_gap_exts_.resize(sz);
    int16_gap_exts_.resize(sz);

    auto gaps_vec = gaps.vec<double>();
    auto gap_ext_vec = gap_exts.vec<double>();
    auto pam_dists_vec = pam_dists.vec<double>();
    auto thresholds_vec = thresholds.vec<double>();

    for (size_t i = 0; i < sz; i++) {
      gaps_.push_back(gaps_vec(i));
      gap_extends_.push_back(gap_ext_vec(i));
      pam_dists_.push_back(pam_dists_vec(i));
      thresholds_.push_back(thresholds_vec(i));
      // parse the tensor protos
      /*OP_REQUIRES(context, double_matrices_t_[i].FromProto(double_matrices[i]),
                  errors::Internal("failed to parse tensorproto"));
      OP_REQUIRES(context, int8_matrices_t_[i].FromProto(int8_matrices[i]),
                  errors::Internal("failed to parse tensorproto"));
      OP_REQUIRES(context, int16_matrices_t_[i].FromProto(int16_matrices[i]),
                  errors::Internal("failed to parse tensorproto"));*/

      // this may not be necessary , but the layout MUST be row oriented contiguous
      double_matrices_[i] = new double[MDIM];

      auto td = double_matrices[i].matrix<double>();

      OP_REQUIRES(context, double_matrices[i].dim_size(0) != double_matrices[i].dim_size(1) != DIMSIZE,
          errors::Internal("double matrix was not 26 x 26, was ", to_string(double_matrices[i].dim_size(0)), 
              " x ", to_string(double_matrices[i].dim_size(1))));
      //LOG(INFO) << double_matrices[i].dim_size(0) << " X " << double_matrices[i].dim_size(1);
      for (size_t rows = 0; rows < double_matrices[i].dim_size(0); rows++) {
          for (size_t cols = 0; cols < double_matrices[i].dim_size(1); cols++) {
              // testing ...
              /*if (i == 0) {
                printf("%f ", td(rows, cols));
                if (cols == double_matrices[i].dim_size(1) - 1)
                  printf("\n");
              }*/
              double_matrices_[i][rows*DIMSIZE + cols] = td(rows, cols);
          }
      }
      int8_matrices_[i] = CreateScaled(double_matrices_[i], thresholds_vec(i), gaps_vec(i),
          gap_ext_vec(i), int8_gaps_[i], int8_gap_exts_[i]);
      int16_matrices_[i] = CreateScaled(double_matrices_[i], thresholds_vec(i), gaps_vec(i),
          gap_ext_vec(i), int16_gaps_[i], int16_gap_exts_[i]);
    }

    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                &object_handle_, nullptr));
  }

  double ByteFactor(double* matrix, double threshold) {
    auto min = fabs(*std::min_element(matrix, matrix + MDIM));
    return 255.0f/(threshold + min);
  }

  double ShortFactor(double threshold) {
    return 65535.0 / threshold;
  }

  int8 ScaleByte(double value, double factor) {
    auto ret = ceil(value*factor);
    if (ret < -128) return -128;
    return int8(ret);
  }
  
  int16 ScaleShort(double value, double factor) {
    auto ret = ceil(value*factor);
    if (ret < -32767) return -32767;
    return int16(ret);
  }

  int8* CreateScaled(double* matrix, double threshold, double gap, double gap_ext, 
      int8& gapi8, int8& gap_exti8) {

    auto factor = ByteFactor(matrix, threshold);
    int8* ret = new int8[MDIM];
    gapi8 = ScaleByte(gap, factor);
    gap_exti8 = ScaleByte(gap_ext, factor);
    for (size_t i = 0; i < MDIM; i++) {
      ret[i] = ScaleByte(matrix[i], factor);
    }
    return ret;
  }
  
  int16* CreateScaled(double* matrix, double threshold, double gap, double gap_ext, 
      int16& gapi16, int16& gap_exti16) {

    auto factor = ShortFactor(threshold);
    int16* ret = new int16[MDIM];
    gapi16 = ScaleShort(gap, factor);
    gap_exti16 = ScaleShort(gap_ext, factor);
    for (size_t i = 0; i < MDIM; i++) {
      ret[i] = ScaleShort(matrix[i], factor);
    }
    return ret;
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    if (!object_handle_set_) {
      OP_REQUIRES_OK(ctx, SetObjectHandle(ctx));
    }
    ctx->set_output_ref(0, &mu_, object_handle_.AccessTensor(ctx));
  }

 protected:
  ~AlignmentEnvironmentsOp() override {
    // If the genome object was not shared, delete it.
    if (object_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
      TF_CHECK_OK(cinfo_.resource_manager()->Delete<EnvsContainer>(
          cinfo_.container(), cinfo_.name()));
    }
  }

 protected:
  ContainerInfo cinfo_;

 private:
  Status SetObjectHandle(OpKernelContext* ctx)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
    EnvsContainer* object;

    auto creator = [this, ctx](EnvsContainer** envs) {
      LOG(INFO) << "loading alignment environments ...";
      auto begin = std::chrono::high_resolution_clock::now();

      AlignmentEnvironments* new_envs = new AlignmentEnvironments();

      // copies here are due to goofiness of dependencies (C source from 1980-something)
      auto gaps_copy = gaps_;
      gaps_copy.insert(gaps_copy.begin(), 0);  // some dependency needs a 1 based array wtf
      auto gap_exs_copy = gap_extends_;
      gap_exs_copy.insert(gap_exs_copy.begin(), 0);
      auto pams_copy = pam_dists_;
      pams_copy.insert(pams_copy.begin(), 0);
      auto matrix_pointers = double_matrices_;
      matrix_pointers.insert(matrix_pointers.begin(), 0);

      new_envs->CreateDayMatrices(gaps_copy, gap_exs_copy, pams_copy, matrix_pointers); 

      vector<AlignmentEnvironment> all_envs;
      for (size_t i = 0; i < gaps_.size(); i++) {
        AlignmentEnvironment newenv;
        newenv.gap_open = gaps_[i];
        newenv.gap_extend = gap_extends_[i];
        newenv.pam_distance = pam_dists_[i];
        newenv.threshold = thresholds_[i];
        newenv.matrix = double_matrices_[i];
        newenv.gap_open_int8 = int8_gaps_[i];
        newenv.gap_ext_int8 = int8_gap_exts_[i];
        newenv.gap_open_int16 = int16_gaps_[i];
        newenv.gap_ext_int16 = int16_gap_exts_[i];
        newenv.matrix_int8 = int8_matrices_[i];
        newenv.matrix_int16 = int16_matrices_[i];

        all_envs.push_back(newenv);
      }
      AlignmentEnvironment logpamenv;
      logpamenv.gap_open = logpam_gap_;
      logpamenv.gap_extend = logpam_gap_ext_;
      logpamenv.pam_distance = logpam_pam_dist_;
      logpamenv.threshold = logpam_threshold_;
      logpamenv.matrix = logpam_matrix_;
      logpamenv.gap_open_int8 = logpam_int8_gap_;
      logpamenv.gap_ext_int8 = logpam_int8_gap_ext_;
      logpamenv.gap_open_int16 = logpam_int16_gap_;
      logpamenv.gap_ext_int16 = logpam_int16_gap_ext_;
      logpamenv.matrix_int8 = logpam_int8_matrix_;
      logpamenv.matrix_int16 = logpam_int16_matrix_;;

      // create score only env
      AlignmentEnvironment justscoreenv;
      justscoreenv.gap_open = -37.64 + 7.434 * log10(224);
      justscoreenv.gap_extend = -1.3961;
      justscoreenv.pam_distance = 224;
      justscoreenv.threshold = 135.75;
      justscoreenv.matrix = new double[MDIM];
      CreateOrigDayMatrix(logpamenv.matrix, 224, justscoreenv.matrix);
      justscoreenv.matrix_int8 = CreateScaled(justscoreenv.matrix, justscoreenv.threshold, justscoreenv.gap_open,
          justscoreenv.gap_extend, justscoreenv.gap_open_int8, justscoreenv.gap_ext_int8);
      justscoreenv.matrix_int16 = CreateScaled(justscoreenv.matrix, justscoreenv.threshold, justscoreenv.gap_open,
          justscoreenv.gap_extend, justscoreenv.gap_open_int16, justscoreenv.gap_ext_int16);

      /*printf("gap: %.10f\n", justscoreenv.gap_open);
      printf("just score matrix\n");
      for (size_t i = 0; i < DIMSIZE; i++) {
        for (size_t j = 0; j < DIMSIZE; j++) {
          printf("%d ", justscoreenv.matrix_int8[i*DIMSIZE + j]);
        }
        printf("\n");
      }*/

/*

  SSW_Environment ssw_env;
  ssw_env.match = 2;
  ssw_env.mismatch_ssw = 2; 
  ssw_env.gap_open = 37.64 - 7.434 * log10(224);
  ssw_env.gap_extension = 1.3961;
  ssw_env.n = 5;
  ssw_env.s1 = 6748000;
  ssw_env.s2 = 128;
  ssw_env.filter = 0;
  ssw_env.mata = (int8_t*)calloc(25, sizeof(int8_t));
  // int8_t mat = (int8_t*)calloc(25, sizeof(int8_t));  
  ssw_env.mat = ssw_env.mata;
  ssw_env.ref_num = (int8_t*)malloc(ssw_env.s1);
  ssw_env.num = (int8_t*)malloc(ssw_env.s2);
  
  // cout << 335 <<endl;
  

  
  for (ssw_env.l = ssw_env.k = 0; LIKELY(ssw_env.l < 4); ++ssw_env.l) {
      for (ssw_env.m = 0; LIKELY(ssw_env.m < 4); ++ssw_env.m) ssw_env.mata[ssw_env.k++] = ssw_env.l == ssw_env.m ? ssw_env.match : -ssw_env.mismatch_ssw; // weight_match : -weight_mismatch_ssw 
      ssw_env.mata[ssw_env.k++] = 0; // ambiguous base
  }
  // cout << 369 <<endl;
  for (ssw_env.m = 0; LIKELY(ssw_env.m < 5); ++ssw_env.m) ssw_env.mata[ssw_env.k++] = 0; //387,385,384 mata -> mat
  ssw_env.n = 24;
  ssw_env.table = ssw_env.aa_table;
  // table = aa_table;
  ssw_env.mat = ssw_env.mat50;
  */


      new_envs->Initialize(all_envs, logpamenv, justscoreenv);//, ssw_env);

      std::unique_ptr<AlignmentEnvironments> value(new_envs);

      *envs = new EnvsContainer(move(value));
      auto end = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "envs load time is: "
                << ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - begin)
                        .count()) /
                       1000000000.0f;
      return Status::OK();
    };

    TF_RETURN_IF_ERROR(
        cinfo_.resource_manager()->LookupOrCreate<EnvsContainer>(
            cinfo_.container(), cinfo_.name(), &object, creator));

    auto h = object_handle_.AccessTensor(ctx)->flat<string>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    object_handle_set_ = true;
    return Status::OK();
  }

  mutex mu_;
  // string genome_location_;
  PersistentTensor object_handle_ GUARDED_BY(mu_);
  bool object_handle_set_ GUARDED_BY(mu_);
  vector<double> gaps_, thresholds_, gap_extends_, pam_dists_;
  vector<double*> double_matrices_;
  vector<int8*> int8_matrices_;
  vector<int16*> int16_matrices_;
  vector<int8> int8_gaps_, int8_gap_exts_;
  vector<int16> int16_gaps_, int16_gap_exts_;

  double* logpam_matrix_;
  int8* logpam_int8_matrix_;
  int16* logpam_int16_matrix_;
  double logpam_gap_, logpam_gap_ext_, logpam_threshold_, logpam_pam_dist_;
  int8 logpam_int8_gap_, logpam_int8_gap_ext_;
  int16 logpam_int16_gap_, logpam_int16_gap_ext_;
};

REGISTER_KERNEL_BUILDER(Name("AlignmentEnvironments").Device(DEVICE_CPU),
                        AlignmentEnvironmentsOp);
}  // namespace tensorflow
