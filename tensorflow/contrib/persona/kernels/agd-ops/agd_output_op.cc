#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"

#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  class AGDOutputOp : public OpKernel {
  public:
    AGDOutputOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("unpack", &unpack_));
      OP_REQUIRES_OK(context, context->GetAttr("columns", &columns_));
      buffers_.resize(columns_.size());
      mmaps_.resize(columns_.size());
      ordinals_.resize(columns_.size());
      num_records_.resize(columns_.size());
      readers_.resize(columns_.size());
    }

    Status LoadChunk(OpKernelContext* ctx, string chunk_path) {

      LOG(INFO) << "chunk path is " << chunk_path;
      for (int i = 0; i < columns_.size(); i++) {

        TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile(path_ + 
              chunk_path + "." + columns_[i], &mmaps_[i]));
        buffers_[i].reset();
        auto unpack = columns_[i] == "base" && unpack_;
        TF_RETURN_IF_ERROR(rec_parser_.ParseNew((const char*)mmaps_[i]->data(), mmaps_[i]->length(),
            false, &buffers_[i], &ordinals_[i], &num_records_[i], record_id_, unpack));
        readers_[i].reset(new AGDRecordReader(buffers_[i].data(), num_records_[i]));
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

      Alignment agd_result;

      while (current <= end) {
        int chunk_offset = current - chunksize*which_chunk;
        if (chunk_offset >= chunksize) { // out of range, load next chunk
          which_chunk++;
          OP_REQUIRES_OK(ctx, LoadChunk(ctx, chunk_names(which_chunk)));
          continue;
        }

        for (int i = 0; i < columns_.size(); i++) {
          if (columns_[i] == "base" || columns_[i] == "metadata" || columns_[i] == "qual") {
            OP_REQUIRES_OK(ctx, readers_[i]->GetRecordAt(chunk_offset, &data, &length));
            fwrite(data, length, 1, stdout);
            printf("\n");
          } else if (columns_[i] == "results" ) {
            OP_REQUIRES_OK(ctx, readers_[i]->GetRecordAt(chunk_offset, &data, &length));
            LOG(INFO) << "length is " << length;
            agd_result.ParseFromArray(data, length);
            printf("Loc: %lld contig: %lld Flag: %04x MAPQ: %d Nextlog: %lld Nextcontig: %lld\n", agd_result.position().position(),
                   agd_result.position().ref_index(), agd_result.flag(),
                   agd_result.mapping_quality(), agd_result.next_position().position(), agd_result.next_position().ref_index());
            printf("CIGAR: %s \n\n", agd_result.cigar().c_str());
          } else if (columns_[i].find("secondary") != std::string::npos) {
            OP_REQUIRES_OK(ctx, readers_[i]->GetRecordAt(chunk_offset, &data, &length));
            printf("Secondary result %d:\n", int(columns_[i].back() - '0'));
            if (length > 0) {
              if (!agd_result.ParsePartialFromArray(data, length))
                LOG(INFO) << "parsing secondary returned false!, length was " << length;
              printf("Loc: %lld contig: %lld Flag: %04x MAPQ: %d Nextloc: %lld Nextcontig: %lld\n", agd_result.position().position(),
                     agd_result.position().ref_index(), agd_result.flag(),
                     agd_result.mapping_quality(), agd_result.next_position().position(), agd_result.next_position().ref_index());
              printf("CIGAR: %s \n\n", agd_result.cigar().c_str());
            } else {
              printf("had an empty secondary result\n");
            }
          } else {
            LOG(INFO) << "Whoops, I don't know what to do for a column named: " << columns_[i];
          }
        }

        current++;
      }
      for (int i = 0; i < columns_.size(); i++) 
        mmaps_[i].reset(nullptr);

    }

  private:
    vector<std::unique_ptr<ReadOnlyMemoryRegion>> mmaps_;

    //Buffer bases_buf_, qual_buf_, meta_buf_, results_buf_;
    vector<Buffer> buffers_;
    //uint64_t bases_ord_, qual_ord_, meta_ord_, results_ord_;
    vector<uint64_t> ordinals_;
    //uint32_t num_bases_, num_qual_, num_meta_, num_results_;
    vector<uint32_t> num_records_;
    vector<unique_ptr<AGDRecordReader>> readers_;

    RecordParser rec_parser_;
    string path_, record_id_;
    bool unpack_;
    vector<string> columns_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDOutput").Device(DEVICE_CPU), AGDOutputOp);
} //  namespace tensorflow {
