#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "format.h"
#include "compression.h"
#include "parser.h"
#include "buffer.h"
#include <vector>
#include <cstdint>
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"

namespace tensorflow {

  REGISTER_OP("DenseReader")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("verify: bool = false")
  .Attr("reserve: int32 = 8192")
  .Input("buffer_pool: Ref(string)")
  .Input("file_handle: string")
  .Output("processed_buffers: string")
  .Output("num_records: int32")
  .Output("first_ordinal: int64")
  .SetIsStateful()
  .Doc(R"doc(
Read in the dense format from an upstream source (file reader or network reader).

Outputs a handle to the buffer containing the processed data

Input buffer_pool is a handle to a tensorflow::BufferPoolOp result tensor,
and file_handle should come from a file_mmap_op

reserve: the number of bytes to call 'reserve' on the vector.
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

      int32_t i;
      OP_REQUIRES_OK(context, context->GetAttr("reserve", &i));
      reserve_bytes_ = static_cast<decltype(reserve_bytes_)>(i);
    }

    ~DenseReaderOp() {
      core::ScopedUnref unref_pool(buffer_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buffer_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pool", &buffer_pool_));
      }

      const Tensor *fileset;
      OP_REQUIRES_OK(ctx, ctx->input("file_handle", &fileset));
      auto fileset_matrix = fileset->matrix<string>();

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));
      auto rmgr = cinfo.resource_manager();

      Tensor *output, *num_records_t, *first_ordinals_t;
      auto fileset_shape = fileset->shape();
      TensorShape vec_shape({fileset_shape.dim_size(0)});
      OP_REQUIRES_OK(ctx, ctx->allocate_output("processed_buffers", fileset->shape(), &output));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records", vec_shape, &num_records_t));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("first_ordinal", vec_shape, &first_ordinals_t));
      auto output_matrix = output->matrix<string>();
      auto num_records = num_records_t->vec<int32>();
      auto first_ordinals = first_ordinals_t->vec<int64>();

      // ALl output is set up at this point

      ResourceContainer<Data> *dense_input;
      ResourceContainer<Buffer> *output_buffer_rc;
      uint64_t first_ord;
      uint32_t num_recs;

      for (int64 i = 0; i < fileset->dim_size(0); i++)
      {
        OP_REQUIRES_OK(ctx, rmgr->Lookup(fileset_matrix(i, 0), fileset_matrix(i, 1), &dense_input));
        core::ScopedUnref unref_me(dense_input);
        ResourceReleaser<Data> dense_releaser(*dense_input);

        OP_REQUIRES_OK(ctx, buffer_pool_->GetResource(&output_buffer_rc));

        auto input_data = dense_input->get();
        auto output_ptr = output_buffer_rc->get();
        auto &output_buffer = output_ptr->get();
        output_buffer.reserve(reserve_bytes_);

        // TODO pass something from from output_file, for parser to fill up
        OP_REQUIRES_OK(ctx, rec_parser_.ParseNew(input_data->data(), input_data->size(),
                                                 verify_, output_buffer, &first_ord, &num_recs));

        output_matrix(i, 0) = output_buffer_rc->container();
        output_matrix(i, 1) = output_buffer_rc->name();

        num_records(i) = num_recs;
        first_ordinals(i) = first_ord;
      }
    }

  private:
    size_t round_ = 0;
    bool verify_ = false;
    RecordParser rec_parser_;
    size_t reserve_bytes_;
    ReferencePool<Buffer> *buffer_pool_ = nullptr;
  };

  REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU), DenseReaderOp);
} //  namespace tensorflow {
