
// Stuart Byma
// Op providing SNAP AlignerOptions

#include <memory>
#include <utility>
#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/PairedAligner.h" // for paired aligner options

namespace tensorflow {
  using namespace std;

  namespace {
    const string options_name("AlignerOptions"), paired_options_name("PairedAlignerOptions");
  }

    template <typename T>
    class AlignerOptionsOp : public OpKernel {
    public:
      typedef BasicContainer<T> OptionsContainer;
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
                OP_REQUIRES_OK(ctx, SetOptionsHandle(ctx));
            }
            ctx->set_output_ref(0, &mu_, options_handle_.AccessTensor(ctx));
        }

    protected:
        ~AlignerOptionsOp() override {
            // If the options object was not shared, delete it.
            if (options_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
                TF_CHECK_OK(cinfo_.resource_manager()->template Delete<OptionsContainer>(
                    cinfo_.container(), cinfo_.name()));
            }
        }

    protected:

        ContainerInfo cinfo_;

    private:
        Status SetOptionsHandle(OpKernelContext* ctx) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
            OptionsContainer* options;

            auto creator = [this](OptionsContainer** options) {
                unique_ptr<T> t(new T("dummy"));
                if (cmd_line_.size() > 0) {
                    LOG(INFO) << "parsing cmd lines ...";
                    const char* argv[cmd_line_.size()];
                    for (size_t i = 0; i < cmd_line_.size(); i++) {
                        argv[i] = cmd_line_[i].c_str();
                    }
                    int argc = cmd_line_.size();
                    bool done;
                    for (int i = 0; i < argc; i++) {
                        if (!t->parse(argv, argc, i, &done))
                            return errors::InvalidArgument("Could not parse snap arg ", string(argv[i]));
                        if (done)
                            break;
                    }
                    LOG(INFO) << "Done!!";
                }

                *options = new OptionsContainer(move(t));
                return Status::OK();
            };

            TF_RETURN_IF_ERROR(
                cinfo_.resource_manager()->template LookupOrCreate<OptionsContainer>(
                    cinfo_.container(), cinfo_.name(), &options, creator));

            auto h = options_handle_.AccessTensor(ctx)->template flat<string>();
            h(0) = cinfo_.container();
            h(1) = cinfo_.name();
            options_handle_set_ = true;
            return Status::OK();
        }

        std::vector<string> cmd_line_;
        mutex mu_;
        PersistentTensor options_handle_ GUARDED_BY(mu_);
        bool options_handle_set_ GUARDED_BY(mu_);
    };


  REGISTER_KERNEL_BUILDER(Name(options_name.c_str()).Device(DEVICE_CPU), AlignerOptionsOp<AlignerOptions>);
  REGISTER_KERNEL_BUILDER(Name(paired_options_name.c_str()).Device(DEVICE_CPU), AlignerOptionsOp<PairedAlignerOptions>);
}  // namespace tensorflow
