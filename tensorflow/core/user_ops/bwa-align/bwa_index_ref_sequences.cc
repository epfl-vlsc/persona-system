#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/user_ops/object-pool/basic_container.h"
#include "bwamem.h"
#include "bwa.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class BwaIndexReferenceSequencesOp : public OpKernel {
    public:
      explicit BwaIndexReferenceSequencesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

      ~BwaIndexReferenceSequencesOp() override {}

      void Compute(OpKernelContext* ctx) override {

        if (index_resource_ == nullptr) {
          OP_REQUIRES_OK(ctx, InitHandles(ctx));
        }

        auto* contigs = bwa_index_->bns->anns;
        auto num_contigs = bwa_index_->bns->n_seqs;

        Tensor* refs_t = NULL, *sizes_t = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({num_contigs}),
              &refs_t));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({num_contigs}),
              &sizes_t));

        auto refs = refs_t->flat<string>();
        auto sizes = sizes_t->flat<int32>();

        for (int i = 0; i < num_contigs; i++) {
          refs(i) = string(contigs[i].name);
          sizes(i) = static_cast<uint32>(contigs[i].len);
          //LOG(INFO) << "contig name is: " << contigs[i].name << " and len is: "
            //<< contigs[i].len;
        }

      }

    private:

      Status InitHandles(OpKernelContext* ctx)
      {
        TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "index_handle", &index_resource_));

        bwa_index_ = index_resource_->get();

        return Status::OK();
      }

      BasicContainer<bwaidx_t> *index_resource_ = nullptr;
      bwaidx_t* bwa_index_ = nullptr;

      TF_DISALLOW_COPY_AND_ASSIGN(BwaIndexReferenceSequencesOp);
  };
  
  using shape_inference::InferenceContext;

  REGISTER_OP("BwaIndexReferenceSequences")
    .Input("index_handle: Ref(string)")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
        return Status::OK();
        })
    .Output("ref_seqs: string")
    .Output("ref_lens: int32")
    .SetIsStateful()
    .Doc(R"doc(
    Given a BWA genome index, produce two vectors containing the contigs
    (ref sequences) and their sizes.
    )doc");

    REGISTER_KERNEL_BUILDER(Name("BwaIndexReferenceSequences").Device(DEVICE_CPU), BwaIndexReferenceSequencesOp);

}  // namespace tensorflow
