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
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"

namespace tensorflow {


  using namespace std;
  using namespace errors;

  class AGDOutputOp : public OpKernel {
  public:
    AGDOutputOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("unpack", &unpack_));
    }

    ~AGDOutputOp() {
    }
  
   
    Status LoadChunk(OpKernelContext* ctx, string chunk_path) {

      LOG(INFO) << "chunk path is " << chunk_path;
      TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile(path_ + chunk_path + ".base", &mmap_bases_));
      bases_buf_.reset();
      TF_RETURN_IF_ERROR(rec_parser_.ParseNew((const char*)mmap_bases_->data(), mmap_bases_->length(),
            false, &bases_buf_, &bases_ord_, &num_bases_, unpack_));
      TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile(path_ + chunk_path + ".qual", &mmap_qual_));
      qual_buf_.reset();
      TF_RETURN_IF_ERROR(rec_parser_.ParseNew((const char*)mmap_qual_->data(), mmap_qual_->length(),
            false, &qual_buf_, &qual_ord_, &num_qual_));
      TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile(path_ + chunk_path + ".metadata", &mmap_meta_));
      meta_buf_.reset();
      TF_RETURN_IF_ERROR(rec_parser_.ParseNew((const char*)mmap_meta_->data(), mmap_meta_->length(),
            false, &meta_buf_, &meta_ord_, &num_meta_));
      Status s = ctx->env()->NewReadOnlyMemoryRegionFromFile(path_ + chunk_path + ".results", &mmap_results_);
      results_buf_.reset();
      if (!s.ok())
        LOG(INFO) << "Results file not found, assuming there are no results for this dataset yet.";
      else {
        TF_RETURN_IF_ERROR(rec_parser_.ParseNew((const char*)mmap_results_->data(), mmap_results_->length(),
              false, &results_buf_, &results_ord_, &num_results_));
      }
      return Status::OK();
    }

    void Compute(OpKernelContext* ctx) override {

      const Tensor *chunk_names_t, *path_t, *start_t, *end_t, *chunk_size_t;
      OP_REQUIRES_OK(ctx, ctx->input("chunk_names", &chunk_names_t));
      OP_REQUIRES_OK(ctx, ctx->input("path", &path_t));
      OP_REQUIRES_OK(ctx, ctx->input("start", &start_t));
      OP_REQUIRES_OK(ctx, ctx->input("finish", &end_t));
      OP_REQUIRES_OK(ctx, ctx->input("chunk_size", &chunk_size_t));
      auto chunk_names = chunk_names_t->vec<string>();
      path_ = path_t->scalar<string>()();
      auto start = start_t->scalar<int>()();
      auto end = end_t->scalar<int>()();
      auto chunksize = chunk_size_t->scalar<int>()();

      Status status;
      const char* data;
      size_t length;
      auto current = start;
      int which_chunk = current / chunksize;

      OP_REQUIRES_OK(ctx, LoadChunk(ctx, chunk_names(which_chunk)));
      AGDRecordReader bases_reader(bases_buf_.data(), chunksize);
      AGDRecordReader qualities_reader(qual_buf_.data(), chunksize);
      AGDRecordReader metadata_reader(meta_buf_.data(), chunksize);
      AGDRecordReader* results_reader = nullptr;
      if (mmap_results_)
        results_reader = new AGDRecordReader(results_buf_.data(), chunksize);

      const format::AlignmentResult* agd_result;

      while (current <= end) {
        int chunk_offset = current - chunksize*which_chunk;
        if (chunk_offset >= chunksize) { // out of range, load next chunk
          which_chunk++;
          OP_REQUIRES_OK(ctx, LoadChunk(ctx, chunk_names(which_chunk)));
          bases_reader = AGDRecordReader(bases_buf_.data(), chunksize);
          qualities_reader = AGDRecordReader(qual_buf_.data(), chunksize);
          metadata_reader = AGDRecordReader(meta_buf_.data(), chunksize);
          if (results_reader)
            *results_reader = AGDRecordReader(results_buf_.data(), chunksize);
          continue;
        }
        OP_REQUIRES_OK(ctx, metadata_reader.GetRecordAt(chunk_offset, &data, &length));
        fwrite(data, length, 1, stdout);
        printf("\n");
        OP_REQUIRES_OK(ctx, bases_reader.GetRecordAt(chunk_offset, &data, &length));
        fwrite(data, length, 1, stdout);
        printf("\n");
        OP_REQUIRES_OK(ctx, qualities_reader.GetRecordAt(chunk_offset, &data, &length));
        fwrite(data, length, 1, stdout);
        printf("\n");
        if (results_reader) {
          OP_REQUIRES_OK(ctx, results_reader->GetRecordAt(chunk_offset, &data, &length));
          agd_result = reinterpret_cast<const format::AlignmentResult*>(data);
          printf("Loc: %lld Flag: %04x MAPQ: %d Next: %ld\n", agd_result->location_, agd_result->flag_, agd_result->mapq_, agd_result->next_location_);
          const char* cigardata = data + sizeof(format::AlignmentResult);
          decltype(length) cigarlen = length - sizeof(format::AlignmentResult);
          fwrite(cigardata, cigarlen, 1, stdout);
          printf("\n\n");
        } else 
          printf("\n");

        current++;
      }
      mmap_bases_.reset(nullptr);
      mmap_qual_.reset(nullptr);
      mmap_meta_.reset(nullptr);
      mmap_results_.reset(nullptr);

    }

  private:
    std::unique_ptr<ReadOnlyMemoryRegion> mmap_bases_;
    std::unique_ptr<ReadOnlyMemoryRegion> mmap_qual_;
    std::unique_ptr<ReadOnlyMemoryRegion> mmap_meta_;
    std::unique_ptr<ReadOnlyMemoryRegion> mmap_results_;
    Buffer bases_buf_, qual_buf_, meta_buf_, results_buf_;
    uint64_t bases_ord_, qual_ord_, meta_ord_, results_ord_;
    uint32_t num_bases_, num_qual_, num_meta_, num_results_;
    RecordParser rec_parser_;
    string path_;
    bool unpack_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDOutput").Device(DEVICE_CPU), AGDOutputOp);
} //  namespace tensorflow {
