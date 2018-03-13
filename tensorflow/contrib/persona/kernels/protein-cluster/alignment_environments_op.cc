
// Stuart Byma
// Op providing Alignment Environments

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
    // OP_REQUIRES_OK(context, context->GetAttr("genome_location",
    // &genome_location_));

    vector<Tensor> double_matrices, int8_matrices, int16_matrices;
    Tensor logpam_matrix, gaps, gap_exts, pam_dists, thresholds;
    OP_REQUIRES_OK(context, context->GetAttr("double_matrices", &double_matrices));
    OP_REQUIRES_OK(context, context->GetAttr("int8_matrices", &int8_matrices));
    OP_REQUIRES_OK(context, context->GetAttr("int16_matrices", &int16_matrices));
    OP_REQUIRES_OK(context, context->GetAttr("gaps", &gaps));
    OP_REQUIRES_OK(context, context->GetAttr("gap_extends", &gap_exts));
    OP_REQUIRES_OK(context, context->GetAttr("pam_dists", &pam_dists));
    OP_REQUIRES_OK(context, context->GetAttr("thresholds", &thresholds));
    //OP_REQUIRES_OK(context, context->GetAttr("logpam_matrix", &logpam_matrix));

   
    auto sz = double_matrices.size();
    LOG(INFO) << "size is " << sz;
    // we can only pass floats and not doubles as attributes what the fuck 
    gaps_.reserve(sz);
    gap_extends_.reserve(sz); 
    pam_dists_.reserve(sz);

    double_matrices_t_.resize(sz);
    int8_matrices_t_.resize(sz);
    int16_matrices_t_.resize(sz);
    double_matrices_.resize(sz);
    int8_matrices_.resize(sz);
    int16_matrices_.resize(sz);

    auto gaps_vec = gaps.vec<double>();
    auto gap_ext_vec = gap_exts.vec<double>();
    auto pam_dists_vec = pam_dists.vec<double>();

    for (size_t i = 0; i < sz; i++) {
      gaps_.push_back(gaps_vec(i));
      gap_extends_.push_back(gap_ext_vec(i));
      pam_dists_.push_back(pam_dists_vec(i));
      // parse the tensor protos
      /*OP_REQUIRES(context, double_matrices_t_[i].FromProto(double_matrices[i]),
                  errors::Internal("failed to parse tensorproto"));
      OP_REQUIRES(context, int8_matrices_t_[i].FromProto(int8_matrices[i]),
                  errors::Internal("failed to parse tensorproto"));
      OP_REQUIRES(context, int16_matrices_t_[i].FromProto(int16_matrices[i]),
                  errors::Internal("failed to parse tensorproto"));*/

      // this may not be necessary , but the layout MUST be row oriented contiguous
      double_matrices_[i] = new double[MDIM];
      int8_matrices_[i] = new int8[MDIM];
      int16_matrices_[i] = new int16[MDIM];

      auto td = double_matrices[i].matrix<double>();
      auto ti8 = int8_matrices[i].matrix<int8>();
      auto ti16 = int16_matrices[i].matrix<int16>();

      OP_REQUIRES(context, double_matrices[i].dim_size(0) != double_matrices[i].dim_size(1) != DIMSIZE,
          errors::Internal("double matrix was not 26 x 26, was ", to_string(double_matrices[i].dim_size(0)), 
              " x ", to_string(double_matrices[i].dim_size(1))));
      OP_REQUIRES(context, int8_matrices[i].dim_size(0) != int8_matrices[i].dim_size(1) != DIMSIZE,
          errors::Internal("int8 matrix was not 26 x 26, was ", to_string(int8_matrices[i].dim_size(0)), 
              " x ", to_string(int8_matrices[i].dim_size(1))));
      OP_REQUIRES(context, int16_matrices[i].dim_size(0) != int16_matrices[i].dim_size(1) != DIMSIZE,
          errors::Internal("int16 matrix was not 26 x 26, was ", to_string(int16_matrices[i].dim_size(0)), 
              " x ", to_string(int16_matrices[i].dim_size(1))));
      //LOG(INFO) << double_matrices[i].dim_size(0) << " X " << double_matrices[i].dim_size(1);
      for (size_t rows = 0; rows < double_matrices[i].dim_size(0); rows++) {
          for (size_t cols = 0; cols < double_matrices[i].dim_size(1); cols++) {
              // testing ...
              if (i == 0) {
                printf("%f ", td(rows, cols));
                if (cols == double_matrices[i].dim_size(1) - 1)
                  printf("\n");
              }
              double_matrices_[i][rows*DIMSIZE + cols] = td(rows, cols);
              int8_matrices_[i][rows*DIMSIZE + cols] = ti8(rows, cols);
              int16_matrices_[i][rows*DIMSIZE + cols] = ti16(rows, cols);
          }
      }
    }

    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                &object_handle_, nullptr));
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


      std::unique_ptr<AlignmentEnvironments> value(new_envs);

      auto end = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "envs load time is: "
                << ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - begin)
                        .count()) /
                       1000000000.0f;
      *envs = new EnvsContainer(move(value));
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
  vector<Tensor> double_matrices_t_, int8_matrices_t_, int16_matrices_t_;
  vector<double> gaps_, thresholds_, gap_extends_, pam_dists_;
  vector<double*> double_matrices_;
  vector<int8*> int8_matrices_;
  vector<int16*> int16_matrices_;
};

REGISTER_KERNEL_BUILDER(Name("AlignmentEnvironments").Device(DEVICE_CPU),
                        AlignmentEnvironmentsOp);
}  // namespace tensorflow
