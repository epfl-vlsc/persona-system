
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

    class BWAIndexOp : public OpKernel {
    public:
      typedef BasicContainer<bwt_t> BWAIndexContainer;

        BWAIndexOp(OpKernelConstruction* context)
            : OpKernel(context), index_handle_set_(false) {
          OP_REQUIRES_OK(context, context->GetAttr("index_location", &index_location_));
          struct stat buf;
          auto ret = stat(index_location_.c_str(), &buf);
          OP_REQUIRES(context, ret == 0 && buf.st_mode & S_IFDIR != 0,
                      Internal("Index location '", index_location_, "' is not a valid directory"));
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
        ~BWAIndexOp() override {
            // If the genome object was not shared, delete it.
            if (genome_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
                TF_CHECK_OK(cinfo_.resource_manager()->Delete<BWAIndexContainer>(
                    cinfo_.container(), cinfo_.name()));
            }
        }

    protected:

        ContainerInfo cinfo_;

    private:
        Status SetIndexHandle(OpKernelContext* ctx, string index_location) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
            BWAIndexContainer* bwa_index;

            auto creator = [this, index_location](BWAIndexContainer** index) {
                LOG(INFO) << "loading bwt index";
                auto begin = std::chrono::high_resolution_clock::now();
                unique_ptr<bwt_t> value(bwt_restore_bwt(index_location_));
                auto end = std::chrono::high_resolution_clock::now();
                LOG(INFO) << "index load time is: " << ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count())/1000000000.0f;
                *index = new BWAIndexContainer(move(value));
                return Status::OK();
            };

            TF_RETURN_IF_ERROR(
                cinfo_.resource_manager()->LookupOrCreate<BWAIndexContainer>(
                    cinfo_.container(), cinfo_.name(), &bwa_index, creator));

            auto h = index_handle_.AccessTensor(ctx)->flat<string>();
            h(0) = cinfo_.container();
            h(1) = cinfo_.name();
            index_handle_set_ = true;
            return Status::OK();
        }

        mutex mu_;
        string index_location_;
        PersistentTensor index_handle_ GUARDED_BY(mu_);
        bool index_handle_set_ GUARDED_BY(mu_);
    };

    REGISTER_OP("BWAIndex")
        .Output("handle: Ref(string)")
        .Attr("index_location: string")
        .Attr("container: string = ''")
        .Attr("shared_name: string = ''")
        .SetIsStateful()
        .Doc(R"doc(
    An op that creates or gives ref to a bwa index.
    handle: The handle to the BWAIndex resource.
    genome_location: The path to the genome index directory.
    container: If non-empty, this index is placed in the given container.
    Otherwise, a default container is used.
    shared_name: If non-empty, this queue will be shared under the given name
    across multiple sessions.
    )doc");

    REGISTER_KERNEL_BUILDER(Name("BWAIndex").Device(DEVICE_CPU), BWAIndexOp);
}  // namespace tensorflow
