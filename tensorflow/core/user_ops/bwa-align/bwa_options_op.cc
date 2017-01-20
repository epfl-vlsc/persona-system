
// Stuart Byma
// Op providing SNAP genome index and genome

#include <sys/stat.h>
#include <sys/types.h>
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
#include "tensorflow/core/user_ops/object-pool/basic_container.h"
#include "bwa/bwt.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

    class BWAOptionsOp : public OpKernel {
    public:
      typedef BasicContainer<mem_opt_t> BWAOptionsContainer;

        BWAOptionsOp(OpKernelConstruction* context)
            : OpKernel(context), index_handle_set_(false) {
          
          OP_REQUIRES_OK(context, context->GetAttr("index_location", &options_));
          
          OP_REQUIRES_OK(context,
                         context->allocate_persistent(DT_STRING, TensorShape({ 2 }),
                                                      &index_handle_, nullptr));
        }

        void Compute(OpKernelContext* ctx) override {
            mutex_lock l(mu_);
            if (!index_handle_set_) {
                OP_REQUIRES_OK(ctx, SetIndexHandle(ctx, index_location_));
            }
            ctx->set_output_ref(0, &mu_, index_handle_.AccessTensor(ctx));
        }

    protected:
        ~BWAOptionsOp() override {
            // If the genome object was not shared, delete it.
            if (genome_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
                TF_CHECK_OK(cinfo_.resource_manager()->Delete<BWAOptionsContainer>(
                    cinfo_.container(), cinfo_.name()));
            }
        }

    protected:

        ContainerInfo cinfo_;

    private:
        Status SetOptionsHandle(OpKernelContext* ctx, string index_location) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
            BWAOptionsContainer* bwa_options;

            auto creator = [this, index_location](BWAOptionsContainer** options) {
                mem_opt_t* opts = mem_opt_init();
                //process opts

                unique_ptr<bwt_t> value(opts);
                *options = new BWAOptionsContainer(move(value));
                return Status::OK();
            };

            TF_RETURN_IF_ERROR(
                cinfo_.resource_manager()->LookupOrCreate<BWAOptionsContainer>(
                    cinfo_.container(), cinfo_.name(), &bwa_options, creator));

            auto h = index_handle_.AccessTensor(ctx)->flat<string>();
            h(0) = cinfo_.container();
            h(1) = cinfo_.name();
            index_handle_set_ = true;
            return Status::OK();
        }

        mutex mu_;
        std::vector<string> options_;
        PersistentTensor index_handle_ GUARDED_BY(mu_);
        bool index_handle_set_ GUARDED_BY(mu_);
    };

    REGISTER_OP("BWAOptions")
        .Output("handle: Ref(string)")
        .Attr("options: list(string)")
        .Attr("container: string = ''")
        .Attr("shared_name: string = ''")
        .SetIsStateful()
        .Doc(R"doc(
    An op that creates or gives ref to a bwa index.
    handle: The handle to the BWAOptions resource.
    genome_location: The path to the genome index directory.
    container: If non-empty, this index is placed in the given container.
    Otherwise, a default container is used.
    shared_name: If non-empty, this queue will be shared under the given name
    across multiple sessions.
    )doc");

    REGISTER_KERNEL_BUILDER(Name("BWAOptions").Device(DEVICE_CPU), BWAOptionsOp);
}  // namespace tensorflow
