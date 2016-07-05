#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "format.h"
#include "compression.h"
#include "parser.h"
#include <vector>
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"

namespace tensorflow {

  REGISTER_OP("DenseReader")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("verify: bool = false")
  .Input("pool_handle: Ref(string)")
  .Input("file_handle: string")
  .Output("record_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Reads the dense stuff
  )doc");

  using namespace std;
  using namespace errors;

  class DenseReaderOp : public OpKernel {
  public:
    DenseReaderOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("verify", &verify_));
      if (verify_) {
        LOG(DEBUG) << name() << " enabled verification\n";
      }
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor *fileset, *parser_pool;
      OP_REQUIRES_OK(ctx, ctx->input("file_handle", &fileset));
      // assume that the python shape function takes care of this
      auto fileset_matrix = fileset->matrix<string>();

      ReferencePool<RecordParser> *ref_pool;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "pool_handle", &ref_pool));
      core::ScopedUnref unref_pool(ref_pool);

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));
      auto rmgr = cinfo.resource_manager();

      Tensor *output;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("record_handle", fileset->shape(), &output));
      auto output_matrix = output->matrix<string>();

      string resource_name(name());
      ResourceContainer<RecordParser> *rec_parser;
      ResourceContainer<Data> *dense_file;
      for (int64 i = 0; i < fileset->dim_size(0); i++)
      {
          OP_REQUIRES_OK(ctx, rmgr->Lookup(fileset_matrix(i, 0), fileset_matrix(i, 1), &dense_file));
          {
            core::ScopedUnref unref_me(dense_file);
            ResourceReleaser<Data> m(*dense_file);

            resource_name = name();
            resource_name.append(to_string(round_++));

            OP_REQUIRES_OK(ctx, ref_pool->GetResource(&rec_parser));

            auto g = dense_file->get();
            OP_REQUIRES_OK(ctx, rec_parser->get()->ParseNew(g->data(), g->size(), verify_, conversion_scratch_, index_scratch_));

            output_matrix(i, 0) = rec_parser->container();
            output_matrix(i, 1) = rec_parser->name();
          }
      }
    }

  private:
    size_t round_ = 0;
    bool verify_ = false;
    // TODO to use for conversion
    vector<char> conversion_scratch_, index_scratch_;
  };

  REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU), DenseReaderOp);
} //  namespace tensorflow {
