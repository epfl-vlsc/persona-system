#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "format.h"
#include "compression.h"
#include "parser.h"
#include "util.h"
#include "buffer.h"
#include <vector>
#include <cstdint>
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"

namespace tensorflow {

  REGISTER_OP("AGDSort")
  .Input("buffer_pool: Ref(string)")
  .Input("results_handles: string")
  .Input("bases_handles: string")
  .Input("qualities_handles: string")
  .Input("metadata_handles: string")
  .Input("num_records: int32")
  .Output("partial_handles: string")
  .SetIsStateful()
  .Doc(R"doc(
Takes N results buffers, and associated bases, qualities and metadata
chunks, and sorts them into a merged a superchunk output buffer. This 
is the main sort step in the AGD external merge sort.

Outputs handles to merged, sorted superchunks in `partial_handles`

Inputs: (N, 2) string handles to buffers containing results, bases,
qualities and metadata.

Currently does not support a general number of columns.

  )doc");

  using namespace std;
  using namespace errors;

  class AGDSortOp : public OpKernel {
  public:
    AGDSortOp(OpKernelConstruction *context) : OpKernel(context) {
    
    }

    ~AGDSortOp() {
      core::ScopedUnref unref_pool(buffer_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buffer_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pool", &buffer_pool_));
      }

      const Tensor *results_in, bases_in, qualities_in, metadata_in, num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->scalar<int32>()();
      OP_REQUIRES_OK(ctx, ctx->input("results_handles", &results_in));
      auto results_matrix = results_in->matrix<string>();
      OP_REQUIRES_OK(ctx, ctx->input("bases_handles", &bases_in));
      auto bases_matrix = bases_in->matrix<string>();
      OP_REQUIRES_OK(ctx, ctx->input("qualities_handles", &qualities_in));
      auto qualities_matrix = qualities_in->matrix<string>();
      OP_REQUIRES_OK(ctx, ctx->input("metadata_handles", &metadata_in));
      auto metadata_matrix = metadata_in->matrix<string>();

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));
      auto rmgr = cinfo.resource_manager();

      Tensor *output;
      TensorShape matrix_shape({4, 2}); // basically a list of (container, name) pairs
      OP_REQUIRES_OK(ctx, ctx->allocate_output("processed_buffers", matrix_shape, &output));
      
      auto output_matrix = output->matrix<string>();

      // phase 1: parse results sequentially, build up vector of (genome_location, index)
      format::AlignmentResult* agd_result;
      auto idx_offset = num_records * sizeof(RelativeIndex);
      auto b = bases->get()->data();
      base_idx_ = reinterpret_cast<const RelativeIndex*>(b);
      base_data_ = b + idx_offset;

      // phase 2: sort the vector by genome_location
      // phase 3: using the sort vector, merge the chunks into superchunks in sorted
      // order

    }

  private:
    RecordParser rec_parser_; // needed?
    ReferencePool<Buffer> *buffer_pool_ = nullptr;
  };

  REGISTER_KERNEL_BUILDER(Name("AGDSort").Device(DEVICE_CPU), AGDSortOp);
} //  namespace tensorflow {
