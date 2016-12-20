#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "data.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include "compression.h"
#include "buffer_list.h"
#include "format.h"
#include "util.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("AGDWriteColumns");
  }

  REGISTER_OP(op_name.c_str())
  .Attr("compress: bool")
  .Attr("record_id: string")
  .Attr("record_type: list({'base', 'qual', 'meta', 'results'})")
  .Attr("output_dir: string = ''")
  .Input("column_handle: string")
  .Input("file_path: string")
  // TODO these can be collapsed into a vec(3) if that would help performance
  .Input("first_ordinal: int64")
  .Input("num_records: int32")
  .Output("key_out: string")
  .Output("first_ordinal_out: int64")
  .SetIsStateful()
  .Doc(R"doc(
Writes out columns from a specified BufferList. The list contains
[data, index] BufferPairs. This Op constructs the header, unifies the buffers,
and writes to disk. Normally, this corresponds to a set of bases, qual, meta, 
results columns. 

This writes out to local disk only

Assumes that the record_id for a given set does not change for the runtime of the graph
and is thus passed as an Attr instead of an input (for efficiency);

This also assumes that this writer only writes out a single record type.
Thus we always need 3 of these for the full conversion pipeline
)doc");

  class AGDWriteColumnsOp : public OpKernel {
  public:
    AGDWriteColumnsOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      using namespace format;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compress", &compress_));
      string rec_id;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_id", &rec_id));
      auto max_size = sizeof(format::FileHeader::string_id);
      OP_REQUIRES(ctx, rec_id.length() < max_size,
                  Internal("record_id for column header '", rec_id, "' greater than 32 characters"));

      vector<string> rec_types;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_type", &rec_types));
      for (size_t i = 0; i < rec_types.size(); ++i) {
        auto& t_in = rec_types[i];
        format::FileHeader header;
        strncpy(header.string_id, rec_id.c_str(), max_size);

        RecordType t;
        if (t_in.compare("base") == 0) {
          t = RecordType::BASES;
        } else if (t_in.compare("qual") == 0) {
          t = RecordType::QUALITIES;
        } else if (t_in.compare("meta") == 0) {
          t = RecordType::COMMENTS;
        } else { // no need to check. we're saved by string enum types if TF
          t = RecordType::ALIGNMENT;
        }
        record_suffixes_.push_back("." + t_in);
        header.record_type = static_cast<uint8_t>(t);
        header.compression_type = compress_ ? CompressionType::GZIP : CompressionType::UNCOMPRESSED;
        headers_.push_back(header);
      }

      string outdir;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dir", &outdir));

      if (!outdir.empty()) {
        struct stat outdir_info;
        OP_REQUIRES(ctx, stat(outdir.c_str(), &outdir_info) == 0, Internal("Unable to stat path: ", outdir));
        OP_REQUIRES(ctx, S_ISDIR(outdir_info.st_mode), Internal("Path ", outdir, " is not a directory"));
      } // else it's just the current working directory
      record_prefix_ = outdir;
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor *path, *column_t;
      OP_REQUIRES_OK(ctx, ctx->input("file_path", &path));
      OP_REQUIRES_OK(ctx, ctx->input("column_handle", &column_t));
      auto filepath = path->scalar<string>()();
      auto column_vec = column_t->vec<string>();

      OP_REQUIRES_OK(ctx, InitHeaders(ctx));

      ResourceContainer<BufferList> *columns;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(column_vec(0), column_vec(1), &columns));
      core::ScopedUnref column_releaser(columns);
      ResourceReleaser<BufferList> a(*columns);

      auto *buf_list = columns->get();
      auto num_buffers = buf_list->size();

      // do this after the wait, so we are only timing the write, and NOT part of the alignment
      //start = chrono::high_resolution_clock::now();
      auto s = Status::OK();

      for (size_t i = 0; i < num_buffers; ++i) {

        string full_path(record_prefix_ + filepath + record_suffixes_[i]);

        FILE *file_out = fopen(full_path.c_str(), "w+");
        // TODO get errno out of file
        OP_REQUIRES(ctx, file_out != nullptr, Internal("Unable to open file at path:", full_path));

        OP_REQUIRES_OK(ctx, WriteHeader(ctx, file_out, i));

        int fwrite_ret;

        if (compress_) {
          OP_REQUIRES(ctx, false, Internal("Compressed out writing for columns not yet supported"));
        } else {
          auto &index = (*buf_list)[i].index();
          auto s1 = index.size();
          if (s1 == 0) {
            OP_REQUIRES(ctx, s1 != 0,
                        Internal("Parallel column writer got an empty index entry (size 0)!"));
          }
          fwrite_ret = fwrite(&index[0], s1, 1, file_out);
          if (fwrite_ret != 1) {
            s = Internal("fwrite (uncompressed) gave non-1 return value: ", fwrite_ret);
            break;
          }
          if (s.ok()) {
            auto &data = (*buf_list)[i].data();

            fwrite_ret = fwrite(&data[0], data.size(), 1, file_out);
            if (fwrite_ret != 1) {
              s = Internal("fwrite (uncompressed) gave non-1 return value: ", fwrite_ret, " when trying to write item of size ", data.size());
              break;
            }
          }
        }

        fclose(file_out);

        if (s.ok() && fwrite_ret != 1) {
          s = Internal("Received non-1 fwrite return value: ", fwrite_ret);
        }
      }

      OP_REQUIRES_OK(ctx, s); // in case s screws up

      Tensor *key_out;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("key_out", TensorShape({}), &key_out));
      key_out->scalar<string>()() = filepath;

      //auto duration = TRACEPOINT_DURATION_CALC(start);
      //tracepoint(bioflow, chunk_read, filepath.c_str(), duration);
    }

  private:

    Status InitHeaders(OpKernelContext *ctx) {
      uint64_t first_ord, last_ord;
      const Tensor *tensor;
      TF_RETURN_IF_ERROR(ctx->input("first_ordinal", &tensor));
      first_ord = static_cast<decltype(first_ord)>(tensor->scalar<int64>()());

      TF_RETURN_IF_ERROR(ctx->input("num_records", &tensor));
      auto num_recs = tensor->scalar<int32>()();
      last_ord = first_ord + static_cast<decltype(last_ord)>(num_recs);

      for (auto &header : headers_) {
        header.first_ordinal = first_ord;
        header.last_ordinal = last_ord;
      }

      Tensor *first_ord_out;
      TF_RETURN_IF_ERROR(ctx->allocate_output("first_ordinal_out", TensorShape({}), &first_ord_out));
      first_ord_out->scalar<int64>()() = first_ord;

      return Status::OK();
    }

    Status WriteHeader(OpKernelContext *ctx, FILE *file_out, int index) {
      auto& header = headers_[index];

      int fwrite_ret = fwrite(&header, sizeof(format::FileHeader), 1, file_out);
      if (fwrite_ret != 1) {
        // TODO get errno out of the file
        fclose(file_out);
        return Internal("fwrite(header_) failed");
      }
      return Status::OK();
    }

    chrono::high_resolution_clock::time_point start;
    bool compress_;
    vector<string> record_suffixes_;
    string record_prefix_;
    vector<char> buf_, outbuf_; // used to compress into
    vector<format::FileHeader> headers_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDWriteColumnsOp);
} //  namespace tensorflow {
