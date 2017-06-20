#include "agd_file_system_writer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  class AGDFileSystemBufferPairWriter : public AGDFileSystemWriterBase {
  public:
    AGDFileSystemBufferPairWriter(OpKernelConstruction *ctx) : AGDFileSystemWriterBase(ctx) {}

  protected:
    Status WriteResource(OpKernelContext *ctx, FILE *f, const std::string &container, const std::string &name) override {
      ResourceContainer<BufferPair> *column;
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, name, &column));

      core::ScopedUnref column_releaser(column);
      {
        ResourceReleaser<BufferPair> pool_releaser(*column);
        auto *bp = column->get();
        auto &buf_pair = *bp;
        auto &index = buf_pair.index();
        auto &data = buf_pair.data();
        TF_RETURN_IF_ERROR(WriteData(f, &index[0], index.size()));
        if (data.size() != 0) {// an emtpy column
          TF_RETURN_IF_ERROR(WriteData(f, &data[0], data.size()));
        } else
          LOG(INFO) << "buffer pair data size was 0!!";
        buf_pair.reset();
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDFileSystemBufferPairWriter").Device(DEVICE_CPU), AGDFileSystemBufferPairWriter);
} // namespace tensorflow {
