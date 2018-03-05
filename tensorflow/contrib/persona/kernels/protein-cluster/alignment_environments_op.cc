
// Stuart Byma
// Op providing Alignment Environments

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/agd-ops/agd_reference_genome.h"

namespace tensorflow {
using namespace std;
using namespace errors;

class AlignmentEnvironmentsOp : public OpKernel {
 public:
  typedef BasicContainer<AlignmentEnvironments> EnvsContainer;

  AlignmentEnvironmentsOp(OpKernelConstruction* context)
      : OpKernel(context), object_handle_set_(false) {
    // OP_REQUIRES_OK(context, context->GetAttr("genome_location",
    // &genome_location_));

    OP_REQUIRES_OK(context, context->GetAttr("double_matrices", &double_matrices_));
    OP_REQUIRES_OK(context, context->GetAttr("int8_matrices", &int8_matrices_));
    OP_REQUIRES_OK(context, context->GetAttr("int16_matrices", &int16_matrices_));
    OP_REQUIRES_OK(context, context->GetAttr("gaps", &gaps_));
    OP_REQUIRES_OK(context, context->GetAttr("gap_extends", &gap_extends_));
    OP_REQUIRES_OK(context, context->GetAttr("thresholds", &thresholds_));

    double_matrices_t_.resize(gaps_.size());
    int8_matrices_t_.resize(gaps_.size());
    int16_matrices_t_.resize(gaps_.size());

    for (size_t i = 0; i < gaps_.size(); i++) {
      OP_REQUIRES(context, double_matrices_t_[i].FromProto(double_matrices_),
                  errors::Internal("failed to parse tensorproto"));
      OP_REQUIRES(context, int8_matrices_t_[i].FromProto(int8_matrices_),
                  errors::Internal("failed to parse tensorproto"));
      OP_REQUIRES(context, int16_matrices_t_[i].FromProto(int16_matrices_),
                  errors::Internal("failed to parse tensorproto"));
    }

    OP_REQUIRES_OK(context,
                   context->allocate_persistent(DT_STRING, TensorShape({2}),
                                                &object_handle_, nullptr));
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(mu_);
    if (!object_handle_set_) {
      OP_REQUIRES_OK(ctx, SetGenomeHandle(ctx, chunk_paths_));
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
  Status SetGenomeHandle(OpKernelContext* ctx, vector<string>& chunk_paths)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
    EnvsContainer* reference_genome;

    auto creator = [this, chunk_paths, ctx](EnvsContainer** envs) {
      LOG(INFO) << "loading alignment environments ...";
      auto begin = std::chrono::high_resolution_clock::now();


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
            cinfo_.container(), cinfo_.name(), &reference_genome, creator));

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
  vector<TensorProto> double_matrices_, int8_matrices_, int16_matrices_;
  vector<Tensor> double_matrices_t_, int8_matrices_t_, int16_matrices_t_;
  vector<double> gaps_, thresholds_, gap_extends_;
};

REGISTER_KERNEL_BUILDER(Name("AlignmentEnvironments").Device(DEVICE_CPU),
                        AlignmentEnvironmentsOp);
}  // namespace tensorflow
