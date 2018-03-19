
// Stuart Byma
// Op providing SNAP genome index and genome

#include <unistd.h>
#include <memory>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/GenomeIndex.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

    class GenomeIndexOp : public OpKernel {
    public:
      typedef BasicContainer<GenomeIndex> GenomeContainer;

        GenomeIndexOp(OpKernelConstruction* context)
            : OpKernel(context), genome_handle_set_(false) {
          OP_REQUIRES_OK(context, context->GetAttr("genome_location", &genome_location_));
          OP_REQUIRES_OK(context, context->env()->FileExists(genome_location_));
          OP_REQUIRES_OK(context,
                         context->allocate_persistent(DT_STRING, TensorShape({ 2 }),
                                                      &genome_handle_, nullptr));
        }

        void Compute(OpKernelContext* ctx) override {
            mutex_lock l(mu_);
            if (!genome_handle_set_) {
                OP_REQUIRES_OK(ctx, SetGenomeHandle(ctx, genome_location_));
            }
            ctx->set_output_ref(0, &mu_, genome_handle_.AccessTensor(ctx));
        }

    protected:
        ~GenomeIndexOp() override {
            // If the genome object was not shared, delete it.
            if (genome_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
                TF_CHECK_OK(cinfo_.resource_manager()->Delete<GenomeContainer>(
                    cinfo_.container(), cinfo_.name()));
            }
        }

    protected:

        ContainerInfo cinfo_;

    private:
        Status SetGenomeHandle(OpKernelContext* ctx, string genome_location) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
            GenomeContainer* genome_index;

            auto creator = [this, genome_location](GenomeContainer** genome) {
                LOG(INFO) << "loading genome index";
                auto begin = std::chrono::high_resolution_clock::now();
                unique_ptr<GenomeIndex> value(GenomeIndex::loadFromDirectory(const_cast<char*>(genome_location.c_str()), true, true));
                auto end = std::chrono::high_resolution_clock::now();
                LOG(INFO) << "genome load time is: " << ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count())/1000000000.0f;
                *genome = new GenomeContainer(move(value));
                return Status::OK();
            };

            TF_RETURN_IF_ERROR(
                cinfo_.resource_manager()->LookupOrCreate<GenomeContainer>(
                    cinfo_.container(), cinfo_.name(), &genome_index, creator));

            auto h = genome_handle_.AccessTensor(ctx)->flat<string>();
            h(0) = cinfo_.container();
            h(1) = cinfo_.name();
            genome_handle_set_ = true;
            return Status::OK();
        }

        mutex mu_;
        string genome_location_;
        PersistentTensor genome_handle_ GUARDED_BY(mu_);
        bool genome_handle_set_ GUARDED_BY(mu_);
    };

    REGISTER_KERNEL_BUILDER(Name("GenomeIndex").Device(DEVICE_CPU), GenomeIndexOp);
}  // namespace tensorflow
