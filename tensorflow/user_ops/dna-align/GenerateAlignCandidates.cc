#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "SnapAlignerWrapper.h"

namespace tensorflow {
    class AlignerResource; // defined below

    class GenerateAlignCandidatesOp : public OpKernel {
    public:
        explicit GenerateAlignCandidatesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("genome_index_path_", &genome_index_path_));

            options_ = new snap_wrapper::AlignmentOptions();
            // TODO configure options here
        }

        ~GenerateAlignCandidatesOp() override {
            if (aligner_) aligner_->Unref();
        }

        void Compute(OpKernelContext* ctx) override {
            {
                mutex_lock l(init_mu_);

                if (aligner_ == nullptr) {
                    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def()));
                    OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<AlignerResource>(
                        cinfo_.container(),
                        cinfo_.name(),
                        &aligner_,
                        [this](AlignerResource** aligner) {
                            *aligner = new AlignerResource();
                            (*aligner)->init(genome_index_path_, options_);
                            return Status::OK();
                        }
                    ));
                }
            }

            const Tensor* read;

            // get the input tensor called "read"
            OP_REQUIRES_OK(ctx, ctx->input("read", &read));

            OpOutputList output;
            OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));

            // allocate output tensor[0]
            // cannot allocate output tensor[1] yet because we 
            // don't know it's dims which will be [num_candidates]
            Tensor* out = nullptr;
            output.allocate(0, TensorShape(), &out);

            // copy over the read, should share same underlying storage as per tensor.h
            OP_REQUIRES(ctx, out->CopyFrom(*read, read->shape()),
                errors::InvalidArgument("GenerateAlignCandidates copy failed, input shape was ",
                    read->shape().DebugString()));
        }

    private:
        // Attributes
        string genome_index_path_;

        // Options
        snap_wrapper::AlignmentOptions* options_;

        // Resources
        mutex init_mu_;
        ContainerInfo cinfo_ GUARDED_BY(init_mu_);
        AlignerResource* aligner_ GUARDED_BY(init_mu_) = nullptr;
    };


    class AlignerResource : public ResourceBase {
    public:
        explicit AlignerResource() {}

        BaseAligner* value() { return value_; }

        void init(string path, snap_wrapper::AlignmentOptions* options) {
            value_ = snap_wrapper::createAligner(snap_wrapper::loadIndex(path.c_str()), options);
        }

        string DebugString() override {
            return "SNAP aligner";
        }

    private:
        BaseAligner* value_;

        TF_DISALLOW_COPY_AND_ASSIGN(AlignerResource);
    };


    REGISTER_OP("GenerateAlignCandidates")
        .Attr("genome_index_location: string")
        .Input("read: string")
        .Output("output: OUT_TYPE")
        .Attr("OUT_TYPE: list({string}) >= 2")
        .Attr("container: string = 'gac_container'")
        .Attr("shared_name: string = 'gac_name'")
        .Doc(R"doc(
Generates a set of alignment candidates for input `read`.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
output: a list of tensors containing the read (0) and a [1] tensor 
containing the alignment candidates. 
)doc");


    REGISTER_KERNEL_BUILDER(Name("GenerateAlignCandidates").Device(DEVICE_CPU), GenerateAlignCandidatesOp);
}