#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "data.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
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

      OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dir", &s));
      if (!s.empty()) {
        record_prefix_ = s;
      }
    }

    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      const Tensor *path, *column_t;
      OP_REQUIRES_OK(ctx, ctx->input("file_path", &path));
      OP_REQUIRES_OK(ctx, ctx->input("column_handle", &column_t));
      auto filepath = path->scalar<string>()();
      auto column_vec = column_t->vec<string>();

      ResourceContainer<BufferList> *column;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(column_vec(0), column_vec(1), &column));
      core::ScopedUnref column_releaser(column);
      ResourceReleaser<BufferList> a(*column);

      auto &buffers = column->get()->get();

      string full_path(record_prefix_ + filepath + record_suffix_);

      FILE *file_out = fopen(full_path.c_str(), "wb");
      // TODO get errno out of file
      OP_REQUIRES(ctx, file_out != NULL,
                  Internal("Unable to open file at path:", full_path));

      OP_REQUIRES_OK(ctx, WriteHeader(ctx, file_out));

      auto num_buffers = buffers.size();
      uint32_t num_records = header_.last_ordinal - header_.first_ordinal;
      uint32_t records_per_chunk = num_records / num_buffers;
      if (num_records % num_buffers != 0) {
        ++records_per_chunk;
      }

      int fwrite_ret;
      auto s = Status::OK();

      decltype(num_records) i = 0, recs_per_chunk = records_per_chunk;
      if (compress_) {
        buf_.clear(); outbuf_.clear();
        /*
        // compressGZIP already calls buf_.clear()
        s = compressGZIP(data->data(), data->size(), buf_);
        if (s.ok()) {
          fwrite_ret = fwrite(buf_.data(), buf_.size(), 1, file_out);
        }
        */
        for (auto &buffer : buffers) {
          if (i + recs_per_chunk > num_records) {
            recs_per_chunk = num_records - i;
          }

          auto &data_buf = buffer.get_when_ready(); // only need to do this on the first call
          s = appendSegment(&data_buf[0], recs_per_chunk, buf_, true);
          if (!s.ok())
            break;

          i += recs_per_chunk;
        }
        if (s.ok()) {
          i = 0; recs_per_chunk = records_per_chunk;
          size_t expected_size;
          for (auto &buffer : buffers) {
            if (i + recs_per_chunk > num_records) {
              recs_per_chunk = num_records - i;
            }

            auto &data_buf = buffer.get();

            expected_size = data_buf.size() - recs_per_chunk;
            s = appendSegment(&data_buf[recs_per_chunk], expected_size, buf_, true);
            if (!s.ok())
              break;
            i += recs_per_chunk;
          }

          if (s.ok()) {
            s = compressGZIP(&buf_[0], buf_.size(), outbuf_);
            if (s.ok()) {
              fwrite_ret = fwrite(&outbuf_[0], outbuf_.size(), 1, file_out);
              if (fwrite_ret != 1) {
                s = Internal("fwrite(compressed) return non-1 value of ", fwrite_ret);
              }
            }
          }
        }
      } else {
        for (auto &buffer : buffers) {
          if (i + recs_per_chunk > num_records) {
            recs_per_chunk = num_records - i;
          }

          auto &data_buf = buffer.get_when_ready();

          fwrite_ret = fwrite(&data_buf[0], recs_per_chunk, 1, file_out);
          if (fwrite_ret != 1) {
            s = Internal("fwrite (uncompressed) gave non-1 return value: ", fwrite_ret);
            break;
          }

          i += recs_per_chunk;
        }
        if (s.ok()) {
          i = 0; recs_per_chunk = records_per_chunk;
          size_t expected_size;
          for (auto &buffer : buffers) {
            if (i + recs_per_chunk > num_records) {
              recs_per_chunk = num_records - i;
            }

            auto &data_buf = buffer.get();

            expected_size = data_buf.size() - recs_per_chunk;
            fwrite_ret = fwrite(&data_buf[recs_per_chunk], expected_size, 1, file_out);
            if (fwrite_ret != 1) {
              s = Internal("fwrite (uncompressed) gave non-1 return value: ", fwrite_ret, " when trying to write item of size ", expected_size);
              break;
            }

            i += recs_per_chunk;
          }
        }
      }

      fclose(file_out);

      if (s.ok() && fwrite_ret != 1) {
        s = Internal("Received non-1 fwrite return value: ", fwrite_ret);
      }

      OP_REQUIRES_OK(ctx, s); // in case s screws up
    }

    ~ParallelColumnWriterOp() {
      LOG(DEBUG) << "parallel column writer " << this << " finishing\n";
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

    bool compress_;
    string record_suffix_, record_prefix_;
    vector<char> buf_, outbuf_; // used to compress into
    format::FileHeader header_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), ParallelColumnWriterOp);
} //  namespace tensorflow {
