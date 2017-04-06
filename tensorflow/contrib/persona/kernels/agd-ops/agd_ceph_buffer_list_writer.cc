#include "agd_ceph_writer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  class AGDCephBufferListWriter : public AGDCephWriterBase {
  public:
    AGDCephBufferListWriter(OpKernelConstruction *ctx) : AGDCephWriterBase(ctx) {}

  protected:
    Status WritePayload(OpKernelContext *ctx, const string &container, const string &name, const string &key, librados::bufferlist &write_buf_list) override {
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
          write_buf_list.push_back(ceph::buffer::create_static(index.size(), const_cast<char*>(&index[0])));
        }
        for (size_t i = 0; i < num_buffers; ++i) {
          auto &data = buf_list[i].data();
          write_buf_list.push_back(ceph::buffer::create_static(data.size(), const_cast<char*>(&data[0])));
        }
        TF_RETURN_IF_ERROR(SendWrite(write_buf_list, key));
        buf_list.reset();
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDCephBufferListWriter").Device(DEVICE_CPU), AGDCephBufferListWriter);
} // namespace tensorflow {
