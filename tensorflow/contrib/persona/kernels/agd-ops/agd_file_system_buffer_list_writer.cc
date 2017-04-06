#include "agd_file_system_writer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  class AGDFileSystemBufferListWriter : public AGDFileSystemWriterBase {
  public:
    AGDFileSystemBufferListWriter(OpKernelConstruction *ctx) : AGDFileSystemWriterBase(ctx) {}

  protected:
    Status WriteResource(OpKernelContext *ctx, FILE *f, const std::string &container, const std::string &name) override {
      ResourceContainer<BufferList> *column;
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, name, &column));

      core::ScopedUnref column_releaser(column);
      {
        ResourceReleaser<BufferList> pool_releaser(*column);
        auto *bl = column->get();
        auto &buf_list = *bl;
        auto num_buffers = bl->size();

        for (size_t i = 0; i < num_buffers; ++i) {
          auto &index = buf_list[i].index();
          TF_RETURN_IF_ERROR(WriteData(f, &index[0], index.size()));
        }
        for (size_t i = 0; i < num_buffers; ++i) {
          auto &data = buf_list[i].data();
          TF_RETURN_IF_ERROR(WriteData(f, &data[0], data.size()));
        }
        buf_list.reset();
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDFileSystemBufferListWriter").Device(DEVICE_CPU), AGDFileSystemBufferListWriter);
} // namespace tensorflow {
