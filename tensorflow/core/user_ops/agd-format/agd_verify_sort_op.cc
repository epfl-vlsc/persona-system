#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "format.h"
#include "column_builder.h"
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

  REGISTER_OP("AGDVerifySort")
  .Input("path: string")
  .Input("chunk_names: string")
  .Input("chunk_size: int32")
  .SetIsStateful()
  .Doc(R"doc(
Verifies that the dataset referred to by `chunk_names` is sorted. 

Chunk names must be in contiguous order.
  )doc");

  using namespace std;
  using namespace errors;

  class AGDVerifySortOp : public OpKernel {
  public:
    AGDVerifySortOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    ~AGDVerifySortOp() {
    }
  
   
    Status LoadChunk(OpKernelContext* ctx, string chunk_path) {

      LOG(INFO) << "chunk path is " << chunk_path;
      TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile(path_ + chunk_path + ".results", &mmap_results_));
      results_buf_.reset();
      TF_RETURN_IF_ERROR(rec_parser_.ParseNew((const char*)mmap_results_->data(), mmap_results_->length(),
            false, &results_buf_, &results_ord_, &num_results_));
      return Status::OK();
    }

    void Compute(OpKernelContext* ctx) override {

      const Tensor *chunk_names_t, *path_t, *chunk_size_t;
      OP_REQUIRES_OK(ctx, ctx->input("chunk_names", &chunk_names_t));
      OP_REQUIRES_OK(ctx, ctx->input("path", &path_t));
      OP_REQUIRES_OK(ctx, ctx->input("chunk_size", &chunk_size_t));
      auto chunk_names = chunk_names_t->vec<string>();
      path_ = path_t->scalar<string>()();
      auto chunksize = chunk_size_t->scalar<int>()();

      Status status;
      const char* data;
      size_t length;
      int64_t prev_location = 0;
      int index = 0;
      const format::AlignmentResult* agd_result;

      for (int i = 0; i < chunk_names.size(); i++) {

        OP_REQUIRES_OK(ctx, LoadChunk(ctx, chunk_names(i)));
        AGDRecordReader results_reader(results_buf_.data(), chunksize);
        index = 0;

        status = results_reader.GetNextRecord(&data, &length);
        while (status.ok()) {
          agd_result = reinterpret_cast<const format::AlignmentResult*>(data);
          //LOG(INFO) << "AGD location is: " << agd_result->location_;
          if (agd_result->location_ < prev_location) {
            LOG(INFO) << "AGD SET IS NOT SORTED. Offending entry in chunk " << i 
              << " at index " << index << ". Prev: " << prev_location << " Curr: "
              << agd_result->location_;

            return;
          } 
          index++;
          prev_location = agd_result->location_;
          status = results_reader.GetNextRecord(&data, &length);
        }
      }

      LOG(INFO) << "AGD Dataset is sorted.";
      mmap_results_.reset(nullptr);

    }

  private:
    std::unique_ptr<ReadOnlyMemoryRegion> mmap_results_;
    Buffer results_buf_;
    uint64_t results_ord_;
    uint32_t num_results_;
    RecordParser rec_parser_;
    string path_;
    bool unpack_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDVerifySort").Device(DEVICE_CPU), AGDVerifySortOp);
} //  namespace tensorflow {
