#include "agd_file_system_writer.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  AGDFileSystemWriterBase::AGDFileSystemWriterBase(OpKernelConstruction *ctx) : AGDWriterBase(ctx) {}

  void AGDFileSystemWriterBase::Compute(OpKernelContext *ctx) {
    const Tensor *path_t, *resource_t;
    OP_REQUIRES_OK(ctx, ctx->input("path", &path_t));
    OP_REQUIRES_OK(ctx, ctx->input("resource_handle", &resource_t));
    auto path = path_t->scalar<string>()();
    auto resource_vec = resource_t->vec<string>();

    OP_REQUIRES_OK(ctx, SetHeaderValues(ctx));

    //VLOG(INFO) << "opening file with path: " << path;
    FILE *f = fopen(path.c_str(), "w+");
    OP_REQUIRES(ctx, f != nullptr, Internal("Unable to open file at path ", path));

    OP_REQUIRES_OK(ctx, WriteData(f, reinterpret_cast<const char*>(&header_), sizeof(header_)));
    OP_REQUIRES_OK(ctx, WriteResource(ctx, f, resource_vec(0), resource_vec(1)));
    OP_REQUIRES_OK(ctx, SetOutputKey(ctx, path));

    fclose(f);
  }

  Status AGDFileSystemWriterBase::WriteData(FILE *f, const char *data, size_t size) {
      auto ret = fwrite(data, size, 1, f);
      if (ret != 1) {
          fclose(f);
          return Internal("Unable to write file, fwrite return value was ", ret, " with errno: ", errno);
      }
      return Status::OK();
  }
} // namespace tensorflow {
