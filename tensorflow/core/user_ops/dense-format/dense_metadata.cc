#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include <string>

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("DenseMetadata");
  }
  
  REGISTER_OP(op_name.c_str())
  .SetIsStateful()
  .Attr("output_dir: string")
  .Input("num_records: int32")
  .Output("first_ordinal: int64")
  .Output("base_file_path: string")
  .Output("meta_file_path: string")
  .Output("qual_file_path: string")
  .Doc(R"doc()doc");

  class DenseMetadataOp : public OpKernel {
  public:
    DenseMetadataOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dir", &output_dir_));
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->scalar<int32>()();
    }
  private:
    string output_dir_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), DenseMetadataOp);
} // namespace tensorflow {
