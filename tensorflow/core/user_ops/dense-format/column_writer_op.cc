#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/lib/core/errors.h"
#include "data.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include "compression.h"
#include "format.h"

namespace tensorflow {

  REGISTER_OP("ColumnWriter")
  .Attr("compress: bool")
  .Attr("record_id: string")
  .Attr("record_type: {'base', 'qual', 'meta', 'results'}")
  .Input("column_handle: string")
  .Input("file_path: string")
  // TODO these can be collapsed into a vec(3) if that would help performance
  .Input("first_ordinal: int64")
  .Input("num_records: int64")
  .SetIsStateful()
  .Doc(R"doc(
Writes out a column (just a character buffer) to the location specified by the input.

This writes out to local disk only

Assumes that the record_id for a given set does not change for the runtime of the graph
and is thus passed as an Attr instead of an input (for efficiency);

This also assumes that this writer only writes out a single record type.
Thus we always need 3 of these for the full conversion pipeline
)doc");

  using namespace std;
  using namespace errors;

  class ColumnWriterOp : public OpKernel {
  public:
    ColumnWriterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
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
      if (s.compare("base")) {
        t = RecordType::BASES;
      } else if (s.compare("qual")) {
        t = RecordType::QUALITIES;
      } else if (s.compare("meta")) {
        t = RecordType::COMMENTS;
      } else { // no need to check. we're saved by string enum types if TF
        t = RecordType::ALIGNMENT;
      }
      header_.record_type = static_cast<uint8_t>(t);
    }

    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      const Tensor *path;
      OP_REQUIRES_OK(ctx, ctx->input("file_path", &path));
      auto filepath = path->scalar<string>()();

      ResourceContainer<Data> *column;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "column_handle", &column));
      ResourceReleaser<Data> a(*column);

      auto data = column->get();

      FILE *file_out = fopen(filepath.c_str(), "wb");
      // TODO get errno out of file
      OP_REQUIRES(ctx, file_out != NULL,
                  Internal("Unable to open file at path:", filepath));

      OP_REQUIRES_OK(ctx, WriteHeader(ctx, file_out));

      int fwrite_ret;
      auto s = Status::OK();
      // TODO in the future, override with a separate op to avoid an if statement
      if (compress_) {
        // compressGZIP already calls buf_.clear()
        s = compressGZIP(data->data(), data->size(), buf_);
        if (s.ok()) {
          fwrite_ret = fwrite(buf_.data(), buf_.size(), 1, file_out);
        }
      } else {
        fwrite_ret = fwrite(data->data(), data->size(), 1, file_out);
      }

      fclose(file_out);

      if (s.ok() && fwrite_ret != 1) {
        s = Internal("Received non-1 fwrite return value: ", fwrite_ret);
      }

      OP_REQUIRES_OK(ctx, s); // in case s screws up
    }

  private:

    Status WriteHeader(OpKernelContext *ctx, FILE *file_out) {
      // TODO get all the incoming parameters for the chunk and write them out
      const Tensor *tensor;
      uint64_t tmp64;
      TF_RETURN_IF_ERROR(ctx->input("first_ordinal", &tensor));
      tmp64 = static_cast<uint64_t>(tensor->scalar<int64>()());
      header_.first_ordinal = tmp64;

      TF_RETURN_IF_ERROR(ctx->input("num_records", &tensor));
      tmp64 = static_cast<uint64_t>(tensor->scalar<int64>()());
      header_.last_ordinal = header_.first_ordinal + tmp64;

      TF_RETURN_IF_ERROR(ctx->input("record_type", &tensor));
      auto record_type = static_cast<uint8_t>(tensor->scalar<int32>()());
      header_.record_type = record_type;

      int fwrite_ret = fwrite(&header_, sizeof(header_), 1, file_out);
      if (fwrite_ret != 1) {
        // TODO get errno out of the file
        return Internal("frwrite(header_) failed");
      }
      return Status::OK();
    }

    bool compress_;
    vector<char> buf_; // used to compress into
    format::FileHeader header_;
  };
} //  namespace tensorflow {