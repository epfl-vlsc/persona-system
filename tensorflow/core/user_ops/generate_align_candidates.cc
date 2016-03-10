
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

REGISTER_OP("GenerateAlignCandidates")
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

class GenerateAlignCandidatesOp : public OpKernel {
 public:
  explicit GenerateAlignCandidatesOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));

    // other constructor stuff ...
  }
  
  ~GenerateAlignCandidatesOp() override {
    if (genome_index_) genome_index_->Unref();
  }

  void Compute(OpKernelContext* ctx) override {
    
    // genome index shared data initializer
    mutex_lock l(init_mu_);
    if (genome_index_ == nullptr) {
      OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def()));

      auto creator = [this](GenomeIndex** g_index) {
        *g_index = new GenomeIndex();
        int* test = (*g_index)->placeholder();
        *test = 1234;
        return Status::OK();
      };
      OP_REQUIRES_OK(ctx,
                     cinfo_.resource_manager()->LookupOrCreate<GenomeIndex>(
                         cinfo_.container(), cinfo_.name(), &genome_index_, creator));
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
  std::vector<DataType> out_type_;

  class GenomeIndex : public ResourceBase {
   public:
    explicit GenomeIndex(/* whatever required */ ) {}
    mutex* mu() { return &mu_; }
    int* placeholder() {return &placeholder_; }

    string DebugString() override {
      return "This is the genome index resource.";
    }

   private:
    mutex mu_; // not sure if needed
    int placeholder_; // replace with SNAP genome index interface

    TF_DISALLOW_COPY_AND_ASSIGN(GenomeIndex);
  };
  
  mutex init_mu_;
  ContainerInfo cinfo_ GUARDED_BY(init_mu_);
  GenomeIndex* genome_index_ GUARDED_BY(init_mu_) = nullptr;
};

REGISTER_KERNEL_BUILDER(Name("GenerateAlignCandidates").Device(DEVICE_CPU), GenerateAlignCandidatesOp);

}  // namespace tensorflow
