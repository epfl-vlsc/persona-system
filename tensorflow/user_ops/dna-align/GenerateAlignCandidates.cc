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
    class GenomeIndexResource; // defined below

    class GenerateAlignCandidatesOp : public OpKernel {
    public:
        explicit GenerateAlignCandidatesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("genome_index_path_", &genome_index_path_));

            options_ = new snap_wrapper::AlignmentOptions();
            // TODO configure options here
        }

        ~GenerateAlignCandidatesOp() override {
            if (genome_index_) genome_index_->Unref();
        }

        void Compute(OpKernelContext* ctx) override {
            {
                mutex_lock l(init_mu_);

                if (genome_index_ == nullptr) {
                    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def()));
                    OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<GenomeIndexResource>(
                        cinfo_.container(),
                        cinfo_.name(),
                        &genome_index_,
                        [this](GenomeIndexResource** aligner) {
                            *aligner = new GenomeIndexResource();
                            (*aligner)->init(genome_index_path_);
                            return Status::OK();
                        }
                    ));
                }
            }

            // TODO read

            // TODO write
        }

    private:
        // Attributes
        string genome_index_path_;

        // Options
        snap_wrapper::AlignmentOptions* options_;

        // Resources
        mutex init_mu_;
        ContainerInfo cinfo_ GUARDED_BY(init_mu_);
        GenomeIndexResource* genome_index_ GUARDED_BY(init_mu_) = nullptr;
    };


    class GenomeIndexResource : public ResourceBase {
    public:
        explicit GenomeIndexResource() {}

        GenomeIndex* value() { return value_; }

        void init(string path) {
            value_ = snap_wrapper::loadIndex(path.c_str());
        }

        string DebugString() override {
            return "SNAP genome index";
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