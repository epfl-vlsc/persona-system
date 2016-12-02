#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "format.h"
#include "agd_record_reader.h"
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
qualities and metadata. num_records is a vector of int32's with the 
number of records per chunk.

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

    Status LoadDataResources(OpKernelContext* ctx, const Tensor* handles_t, 
        vector<const char*> &vec) {
      auto rmgr = ctx->resource_manager();
      auto handles_matrix = handles_t->matrix<string>();
      auto num = handles_t->shape().dim_size(0);
      ResourceContainer<Data> *input;

      for (int i = 0; i < num; i++) {
        OP_REQUIRES_OK(ctx, rmgr->Lookup(handles_matrix(i, 0), handles_matrix(i, 1), &input));
        vec.push_back(input->get()->data());
      }
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buffer_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pool", &buffer_pool_));
      }

      const Tensor *results_in, bases_in, qualities_in, metadata_in, num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->vector<int32>();
      OP_REQUIRES_OK(ctx, ctx->input("results_handles", &results_in));
      OP_REQUIRES_OK(ctx, ctx->input("bases_handles", &bases_in));
      OP_REQUIRES_OK(ctx, ctx->input("qualities_handles", &qualities_in));
      OP_REQUIRES_OK(ctx, ctx->input("metadata_handles", &metadata_in));

      vector<AGDRecordReader> results_vec;
      LoadDataResources(ctx, results_in, results_vec);

      Tensor *output;
      TensorShape matrix_shape({4, 2}); // basically a list of (container, name) pairs
      OP_REQUIRES_OK(ctx, ctx->allocate_output("processed_buffers", matrix_shape, &output));
      
      auto output_matrix = output->matrix<string>();

      // phase 1: parse results sequentially, build up vector of (genome_location, index)
      format::AlignmentResult* agd_result;
      auto num_results = results_in->shape().dim_size(0);
      for (int i = 0; i < num_results; i++) {
        auto num_recs = num_records(i);
        auto idx_offset = num_recs * sizeof(RelativeIndex);
        auto r = results_vec[i];



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
