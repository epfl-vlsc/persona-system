
// Stuart Byma
// Op providing SNAP AlignerOptions

#ifndef TENSORFLOW_KERNELS_ALIGNEROPTIONSOP_H_
#define TENSORFLOW_KERNELS_ALIGNEROPTIONSOP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/user_ops/dna-align/aligner_options_resource.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

    class AlignerOptionsOp : public OpKernel {
    public:
        AlignerOptionsOp(OpKernelConstruction* context)
            : OpKernel(context), options_handle_set_(false) {
            OP_REQUIRES_OK(context, context->GetAttr("cmd_line", &cmd_line_));
            OP_REQUIRES_OK(context,
                context->allocate_persistent(DT_STRING, TensorShape({ 2 }),
                    &options_handle_, nullptr));
        }

        void Compute(OpKernelContext* ctx) override {
            mutex_lock l(mu_);
            if (!options_handle_set_) {
                OP_REQUIRES_OK(ctx, SetOptionsHandle(ctx, cmd_line_));
            }
            ctx->set_output_ref(0, &mu_, options_handle_.AccessTensor(ctx));
        }

    protected:
        ~AlignerOptionsOp() override {
            // If the options object was not shared, delete it.
            if (options_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
                TF_CHECK_OK(cinfo_.resource_manager()->Delete<AlignerOptionsResource>(
                    cinfo_.container(), cinfo_.name()));
            }
        }

    protected:

        ContainerInfo cinfo_;

    private:
        Status SetOptionsHandle(OpKernelContext* ctx, string cmd_line) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
            AlignerOptionsResource* options;

            auto creator = [this, cmd_line](AlignerOptionsResource** options) {
                *options = new AlignerOptionsResource();
                (*options)->init(cmd_line_);
                return Status::OK();
            };

            TF_RETURN_IF_ERROR(
                cinfo_.resource_manager()->LookupOrCreate<AlignerOptionsResource>(
                    cinfo_.container(), cinfo_.name(), &options, creator));

            auto h = options_handle_.AccessTensor(ctx)->flat<string>();
            h(0) = cinfo_.container();
            h(1) = cinfo_.name();
            options_handle_set_ = true;
            return Status::OK();
        }

        string cmd_line_;
        mutex mu_;
        PersistentTensor options_handle_ GUARDED_BY(mu_);
        bool options_handle_set_ GUARDED_BY(mu_);
    };

    REGISTER_OP("AlignerOptions")
        .Output("handle: Ref(string)")
        .Attr("cmd_line: string")
        .Attr("container: string = ''")
        .Attr("shared_name: string = ''")
        .SetIsStateful()
        .Doc(R"doc(
An op that produces SNAP aligner options.
handle: The handle to the options.
cmd_line: The SNAP command line parsed to create the options.
container: If non-empty, this options is placed in the given container.
        Otherwise, a default container is used.
shared_name: If non-empty, this options will be shared under the given name
  across multiple sessions.
)doc");

    REGISTER_KERNEL_BUILDER(Name("AlignerOptions").Device(DEVICE_CPU), AlignerOptionsOp);
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_QUEUE_OP_H_
