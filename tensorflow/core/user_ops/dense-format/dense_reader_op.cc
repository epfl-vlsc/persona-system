#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "format.h"
#include "decompress.h"
#include "parser.h"
#include <vector>
#include <typeinfo>

namespace tensorflow {

  REGISTER_OP("DenseReader")
  .Attr("batch_size: int")
  .Attr("size_hint: int = 4194304") // 4 MeB
  .Input("file_handle: string")
  .Output("record_handle: int64")
  .SetIsStateful()
  .Doc(R"doc(
Reads the dense stuff
  )doc");

  using namespace std;

  class DenseReaderOp : public OpKernel {
  public:
    DenseReaderOp(OpKernelConstruction *context) : OpKernel(context) {
      using namespace errors;
      int batch_size;
      OP_REQUIRES_OK(context, context->GetAttr("batch_size",
                                               &batch_size));
      OP_REQUIRES(context, batch_size > 0, InvalidArgument("DenseReaderOp: batch_size must be >0 - ", batch_size));
      batch_size_ = batch_size;

      OP_REQUIRES_OK(context, context->GetAttr("size_hint", &batch_size));
      size_hint_ = static_cast<size_t>(batch_size);
      OP_REQUIRES(context, size_hint_ > 0, InvalidArgument("DenseReaderOp: size_hint_ must be >0 - ", size_hint_));
    }

    ~DenseReaderOp() {}

    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      const Tensor *fileset;
      OP_REQUIRES_OK(ctx, ctx->input("file_handle", &fileset));
      OP_REQUIRES(ctx, fileset->shape() == TensorShape({2}), InvalidArgument("Tensorshape is incorrect for dense reader op"));

      ReadOnlyFileRef file_handle(fileset);

      { // for the scoped unref
        MemoryMappedFile *dense_file;
        OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(file_handle.GetContainer(), file_handle.GetName(), &dense_file));
        core::ScopedUnref unref_me(dense_file);
        while (!dense_file->RefCountIsOne()) {
          dense_file->Unref();
        }
        dense_file->Ref(); // what a hack :(

        auto dense_mapping = dense_file->GetMappedRegion();
        auto data_buffer = new RecordParser(size_hint_);

        OP_REQUIRES_OK(ctx, data_buffer->ParseNew(static_cast<const char*>(dense_mapping->data()), dense_mapping->length()));

        // TODO just emit it as a single scalar value
        Tensor *output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("record_handle", TensorShape(), &output));
        auto handle = output->scalar<int64>();
        handle() = reinterpret_cast<int64>(data_buffer);
      }
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Delete<MemoryMappedFile>(file_handle.GetContainer(), file_handle.GetName()));
    }

  private:
    int batch_size_;
    size_t size_hint_;
  };

  REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU), DenseReaderOp);
} //  namespace tensorflow {
