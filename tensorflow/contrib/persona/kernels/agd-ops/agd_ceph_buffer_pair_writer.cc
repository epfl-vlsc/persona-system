#include "agd_ceph_writer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_pair.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  class AGDCephBufferPairWriter : public AGDCephWriterBase {
  public:
    AGDCephBufferPairWriter(OpKernelConstruction *ctx) : AGDCephWriterBase(ctx) {}

  protected:
    Status WritePayload(OpKernelContext *ctx, const string &container, const string &name, const string &key, librados::bufferlist &write_buf_list) override {
      ResourceContainer<BufferPair> *column;
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, name, &column));

      core::ScopedUnref column_releaser(column);
      {
        ResourceReleaser<BufferPair> pool_releaser(*column);
        auto *bp = column->get();
        auto &buf_pair = *bp;
        auto &index = buf_pair.index();
        auto &data = buf_pair.data();
        write_buf_list.push_back(ceph::buffer::create_static(index.size(), const_cast<char*>(&index[0])));
        write_buf_list.push_back(ceph::buffer::create_static(data.size(), const_cast<char*>(&data[0])));
        TF_RETURN_IF_ERROR(SendWrite(write_buf_list, key));
        buf_pair.reset();
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDCephBufferPairWriter").Device(DEVICE_CPU), AGDCephBufferPairWriter);
} // namespace tensorflow {
