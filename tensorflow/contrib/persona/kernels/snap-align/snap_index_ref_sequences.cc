#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/GenomeIndex.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/Genome.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class SnapIndexReferenceSequencesOp : public OpKernel {
    public:
      explicit SnapIndexReferenceSequencesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

      ~SnapIndexReferenceSequencesOp() override {}

      void Compute(OpKernelContext* ctx) override {

        if (index_resource_ == nullptr) {
          OP_REQUIRES_OK(ctx, InitHandles(ctx));
        }

        auto* contigs = genome_->getContigs();
        int num_contigs = genome_->getNumContigs();

        Tensor* refs_t = NULL, *sizes_t = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({num_contigs}),
              &refs_t));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({num_contigs}),
              &sizes_t));

        auto refs = refs_t->flat<string>();
        auto sizes = sizes_t->flat<int32>();

        for (int i = 0; i < num_contigs; i++) {
          refs(i) = string(contigs[i].name, contigs[i].nameLength);
          sizes(i) = static_cast<uint32>(contigs[i].length);
        }

      }

    private:

      Status InitHandles(OpKernelContext* ctx)
      {
        TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "genome_handle", &index_resource_));

        genome_ = index_resource_->get()->getGenome();

        return Status::OK();
      }

      BasicContainer<GenomeIndex> *index_resource_ = nullptr;
      const Genome *genome_ = nullptr;

      TF_DISALLOW_COPY_AND_ASSIGN(SnapIndexReferenceSequencesOp);
  };

  REGISTER_KERNEL_BUILDER(Name("SnapIndexReferenceSequences").Device(DEVICE_CPU), SnapIndexReferenceSequencesOp);

}  // namespace tensorflow
