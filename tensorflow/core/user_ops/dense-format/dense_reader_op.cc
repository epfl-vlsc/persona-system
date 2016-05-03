#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "format.h"
#include "decompress.h"
#include "shared_mmap_file_resource.h"
#include "parser.h"
#include <vector>

namespace tensorflow {

  REGISTER_OP("DenseReader")
  .Attr("batch_size: int")
  .Input("file_handle: string")
  .Output("records: string")
  .Output("record_count: int32")
  .SetIsStateful()
  .Doc(R"doc(
Reads the dense stuff
  )doc");

  using namespace std;

  class DenseReader : public OpKernel {
  public:
    DenseReader(OpKernelConstruction *context) : OpKernel(context) {
      using namespace errors;
      int batch_size;
      OP_REQUIRES_OK(context, context->GetAttr("batch_size",
                                               &batch_size));
      OP_REQUIRES(context, batch_size > 0, InvalidArgument("DenseReaderOp: batch_size must be >0 - ", batch_size));
      batch_size_ = batch_size;
    }


    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      const Tensor *fileset;
      OP_REQUIRES_OK(ctx, ctx->input("file_handle", &fileset));
      OP_REQUIRES(ctx, fileset->shape() == TensorShape({2}), InvalidArgument("Tensorshape is incorrect for dense reader op"));

      ReadOnlyFileRef file_handle(fileset);

      MemoryMappedFile *dense_file;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, file_handle.GetName(), &dense_file));
      core::ScopedUnref unref_me(dense_file);
      auto dense_mapping = dense_file->GetMappedRegion();

      OP_REQUIRES_OK(ctx, data_buffer_.ParseNew(static_cast<const char*>(dense_mapping->data()), dense_mapping->length()));

      auto num_records = data_buffer_.RecordCount();
      OP_REQUIRES(ctx, num_records <= batch_size_,
                  Internal("Record Count ", num_records,
                           " exceeds batch size ", batch_size_));
      Tensor *output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("records", TensorShape({batch_size_}), &output));
      auto flat = output->vec<string>();

      size_t i = 0;
      string s;
      for (; i < num_records; i++) {
        OP_REQUIRES_OK(ctx, data_buffer_.GetNextRecord(&s));
        flat(i) = s;
      }
      for (; i < batch_size_; i++ ) {
        flat(i) = "";
      }

      Tensor *size_tensor = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("record_size", TensorShape({}), &output));
      auto size_tensor_scalar = size_tensor->scalar<int>();
      size_tensor_scalar() = num_records;
    }

  private:

    int batch_size_;
    RecordParser data_buffer_;
  };

  REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU), DenseReader);
} //  namespace tensorflow {
