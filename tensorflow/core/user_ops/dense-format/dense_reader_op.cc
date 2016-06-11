#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "format.h"
#include "decompress.h"
#include "parser.h"
#include <vector>
#include "scope_timer.h"

namespace tensorflow {

  REGISTER_OP("DenseReader")
  .Attr("batch_size: int")
  .Attr("size_hint: int = 4194304") // 4 MeB
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("pool_handle: Ref(string)")
  .Input("file_handle: string")
  .Output("record_handle: string")
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
      OP_REQUIRES(context, size_hint_ > 0, InvalidArgument("DenseReaderOp: size_hint_ must be > 0 - ", size_hint_));
    }

    ~DenseReaderOp() {}

    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      const Tensor *fileset, *parser_pool;
      OP_REQUIRES_OK(ctx, ctx->input("file_handle", &fileset));
      // assume that the python shape function takes care of this
      auto fileset_matrix = fileset->matrix<string>();

      OP_REQUIRES_OK(ctx, ctx->input("pool_handle", &parser_pool));

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));
      auto rmgr = cinfo.resource_manager();

      Tensor *output;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("record_handle", fileset->shape(), &output));
      auto output_matrix = output->matrix<string>();

      string resource_name(name());
      RecordParser *rp;
      MemoryMappedFile *dense_file;
      for (int64 i = 0; i < fileset->dim_size(0); i++)
      {
          OP_REQUIRES_OK(ctx, rmgr->Lookup(fileset_matrix(i, 0), fileset_matrix(i, 1), &dense_file));
          {
            core::ScopedUnref unref_me(dense_file);

            auto dense_mapping = dense_file->GetMappedRegion();

            resource_name = name();
            resource_name.append(to_string(round_++));

            rp = new RecordParser(size_hint_);
            OP_REQUIRES_OK(ctx, rp->ParseNew(static_cast<const char*>(dense_mapping->data()), dense_mapping->length()));
            OP_REQUIRES_OK(ctx, rmgr->Create<RecordParser>(cinfo.container(), resource_name, rp));

            output_matrix(i, 0) = cinfo.container();
            output_matrix(i, 1) = resource_name;
          }
          OP_REQUIRES_OK(ctx, rmgr->Delete<MemoryMappedFile>(fileset_matrix(i, 0), fileset_matrix(i, 1)));
      }
    }

  private:
    int batch_size_;
    size_t size_hint_;
    size_t round_ = 0;
    //WritableFile *decomp;
  };

  REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU), DenseReaderOp);
} //  namespace tensorflow {
