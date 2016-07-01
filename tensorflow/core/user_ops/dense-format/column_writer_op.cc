#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/dna-align/data.h"
#include <cstdio>
#include <vector>
#include "compress.h"

namespace tensorflow {

  REGISTER_OP("ColumnWriter")
  .Attr("compress: bool = true")
  .Input("column_handle: string")
  .Input("file_path: string")
  .SetIsStateful()
  .Doc(R"doc(
Writes out a column (just a character buffer) to the location specified by the input.

This writes out to local disk only
)doc");

  using namespace std;

  class ColumnWriterOp : public OpKernel {
  public:
    ColumnWriterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compress", &compress_));
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

      int fwrite_ret;
      Status s(Status::OK());
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
      OP_REQUIRES_OK(ctx, s); // in case s screws up
    }

  private:
    bool compress_;
    vector<char> buf_; // used to compress into
  };
} //  namespace tensorflow {
