#include <boost/timer/timer.hpp>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "format.h"
#include "decompress.h"
#include "parser.h"
#include "scope_timer.h"
#include <vector>

namespace tensorflow {

  REGISTER_OP("DenseReader")
  .Attr("batch_size: int")
  .Attr("trace_file: string") // only for tracing timing
  .Attr("trace_file_process: string")
  .Attr("trace_file_decomp: string")
  .Input("file_handle: string")
  .Output("records: string")
  .Output("record_count: int32")
  .SetIsStateful()
  .Doc(R"doc(
Reads the dense stuff
  )doc");

  constexpr int64_t ipow(int64_t base, int exp, int64_t result = 1) {
    return exp < 1 ? result : ipow(base*base, exp/2, (exp % 2) ? result*base : result);
  }

  using namespace std;

  class DenseReaderOp : public OpKernel {
  public:
    DenseReaderOp(OpKernelConstruction *context) : OpKernel(context), data_buffer_(536870912) {
      using namespace errors;
      int batch_size;
      OP_REQUIRES_OK(context, context->GetAttr("batch_size",
                                               &batch_size));
      OP_REQUIRES(context, batch_size > 0, InvalidArgument("DenseReaderOp: batch_size must be >0 - ", batch_size));
      batch_size_ = batch_size;

      string trace_file;
      OP_REQUIRES_OK(context, context->GetAttr("trace_file",
                                               &trace_file));
      OP_REQUIRES_OK(context, context->env()->NewWritableFile(trace_file, &trace_file_));
      OP_REQUIRES_OK(context, trace_file_->Append("time,duration\n"));

      OP_REQUIRES_OK(context, context->GetAttr("trace_file_process",
                                               &trace_file));
      OP_REQUIRES_OK(context, context->env()->NewWritableFile(trace_file, &convert_trace_file_));
      OP_REQUIRES_OK(context, convert_trace_file_->Append("time,duration\n"));

      OP_REQUIRES_OK(context, context->GetAttr("trace_file_decomp",
                                               &trace_file));
      OP_REQUIRES_OK(context, context->env()->NewWritableFile(trace_file, &decomp_trace_file_));
      OP_REQUIRES_OK(context, decomp_trace_file_->Append("time,duration\n"));
    }

    ~DenseReaderOp() {
      if (trace_file_)
        delete trace_file_;
    }

    void Compute(OpKernelContext* ctx) override {
      ScopeTimer s(trace_file_);
      using namespace errors;
      const Tensor *fileset;
      OP_REQUIRES_OK(ctx, ctx->input("file_handle", &fileset));
      OP_REQUIRES(ctx, fileset->shape() == TensorShape({2}), InvalidArgument("Tensorshape is incorrect for dense reader op"));

      ReadOnlyFileRef file_handle(fileset);

      { // for the scoped unref
        MemoryMappedFile *dense_file;
        OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(file_handle.GetContainer(), file_handle.GetName(), &dense_file));
        core::ScopedUnref unref_me(dense_file);
        auto dense_mapping = dense_file->GetMappedRegion();

        {
          ScopeTimer x(decomp_trace_file_);
          OP_REQUIRES_OK(ctx, data_buffer_.ParseNew(static_cast<const char*>(dense_mapping->data()), dense_mapping->length()));
        }

        {
          ScopeTimer t(convert_trace_file_);
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
          OP_REQUIRES_OK(ctx, ctx->allocate_output("record_count", TensorShape({}), &size_tensor));
          auto size_tensor_scalar = size_tensor->scalar<int>();
          size_tensor_scalar() = num_records;
          while (!dense_file->RefCountIsOne()) {
            dense_file->Unref();
          }
          dense_file->Ref(); // what a hack :(
        }
      }
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Delete<MemoryMappedFile>(file_handle.GetContainer(), file_handle.GetName()));
    }

  private:
    int batch_size_;
    RecordParser data_buffer_;
    WritableFile *trace_file_ = nullptr, *convert_trace_file_ = nullptr, *decomp_trace_file_ = nullptr;
  };

  REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU), DenseReaderOp);
} //  namespace tensorflow {
