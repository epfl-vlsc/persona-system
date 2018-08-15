#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
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
        auto sizes = sizes_t->flat<string>();

        for (int i = 0; i < num_contigs; i++) {
          refs(i) = string(contigs[i].name);
          sizes(i) = to_string(contigs[i].len);
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

  REGISTER_KERNEL_BUILDER(Name("BwaIndexReferenceSequences").Device(DEVICE_CPU), BwaIndexReferenceSequencesOp);

}  // namespace tensorflow
