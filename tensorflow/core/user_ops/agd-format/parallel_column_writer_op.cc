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
    const string op_name("ParallelColumnWriter");
  }

  REGISTER_OP(op_name.c_str())
  .Attr("compress: bool")
  .Attr("record_id: string")
  .Attr("record_type: {'base', 'qual', 'meta', 'results'}")
  .Attr("output_dir: string = ''")
  .Input("column_handle: string")
  .Input("file_path: string")
  // TODO these can be collapsed into a vec(3) if that would help performance
  .Input("first_ordinal: int64")
  .Input("num_records: int32")
  .Output("key_out: string")
  .SetIsStateful()
  .Doc(R"doc(
Writes out a column (just a character buffer) to the location specified by the input.

This writes out to local disk only

Assumes that the record_id for a given set does not change for the runtime of the graph
and is thus passed as an Attr instead of an input (for efficiency);

This also assumes that this writer only writes out a single record type.
Thus we always need 3 of these for the full conversion pipeline
)doc");

  class ParallelColumnWriterOp : public OpKernel {
  public:
    ParallelColumnWriterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      using namespace format;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compress", &compress_));
      string s;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_id", &s));
      auto max_size = sizeof(header_.string_id);
      OP_REQUIRES(ctx, s.length() < max_size,
                  Internal("record_id for column header '", s, "' greater than 32 characters"));
      strncpy(header_.string_id, s.c_str(), max_size);

      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_type", &s));
      RecordType t;
      if (s.compare("base") == 0) {
        t = RecordType::BASES;
      } else if (s.compare("qual") == 0) {
        t = RecordType::QUALITIES;
      } else if (s.compare("meta") == 0) {
        t = RecordType::COMMENTS;
      } else { // no need to check. we're saved by string enum types if TF
        t = RecordType::ALIGNMENT;
      }
      record_suffix_ = "." + s;
      header_.record_type = static_cast<uint8_t>(t);

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

      ResourceContainer<BufferList> *column;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(column_vec(0), column_vec(1), &column));
      core::ScopedUnref column_releaser(column);
      ResourceReleaser<BufferList> a(*column);

      auto *buf_list = column->get();
      buf_list->wait_for_ready();

      // do this after the wait, so we are only timing the write, and NOT part of the alignment
      start = chrono::high_resolution_clock::now();

      string full_path(record_prefix_ + filepath + record_suffix_);

      FILE *file_out = fopen(full_path.c_str(), "w+");
      // TODO get errno out of file
      OP_REQUIRES(ctx, file_out != NULL,
                  Internal("Unable to open file at path:", full_path));

      OP_REQUIRES_OK(ctx, WriteHeader(ctx, file_out));

      auto num_buffers = buf_list->size();
      size_t i;

      int fwrite_ret;
      auto s = Status::OK();

      if (compress_) {
        OP_REQUIRES(ctx, false, Internal("Compressed out writing for columns not yet supported"));
      } else {
        for (i = 0; i < num_buffers; ++i) {
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
        }
        if (s.ok()) {
          for (i = 0; i < num_buffers; ++i) {
            auto &data = (*buf_list)[i].data();

            fwrite_ret = fwrite(&data[0], data.size(), 1, file_out);
            if (fwrite_ret != 1) {
              s = Internal("fwrite (uncompressed) gave non-1 return value: ", fwrite_ret, " when trying to write item of size ", data.size());
              break;
            }
          }
        }
      }

      fclose(file_out);

      if (s.ok() && fwrite_ret != 1) {
        s = Internal("Received non-1 fwrite return value: ", fwrite_ret);
      }


      OP_REQUIRES_OK(ctx, s); // in case s screws up

      Tensor *key_out;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("key_out", TensorShape({}), &key_out));
      key_out->scalar<string>()() = filepath;

      auto duration = TRACEPOINT_DURATION_CALC(start);
      tracepoint(bioflow, chunk_read, filepath.c_str(), duration);
    }

  private:

    Status WriteHeader(OpKernelContext *ctx, FILE *file_out) {
      const Tensor *tensor;
      uint64_t tmp64;
      TF_RETURN_IF_ERROR(ctx->input("first_ordinal", &tensor));
      tmp64 = static_cast<decltype(tmp64)>(tensor->scalar<int64>()());
      header_.first_ordinal = tmp64;

      TF_RETURN_IF_ERROR(ctx->input("num_records", &tensor));
      tmp64 = static_cast<decltype(tmp64)>(tensor->scalar<int32>()());
      header_.last_ordinal = header_.first_ordinal + tmp64;

      int fwrite_ret = fwrite(&header_, sizeof(header_), 1, file_out);
      if (fwrite_ret != 1) {
        // TODO get errno out of the file
        fclose(file_out);
        return Internal("frwrite(header_) failed");
      }
      return Status::OK();
    }

    chrono::high_resolution_clock::time_point start;
    bool compress_;
    string record_suffix_, record_prefix_;
    vector<char> buf_, outbuf_; // used to compress into
    format::FileHeader header_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), ParallelColumnWriterOp);
} //  namespace tensorflow {
