#include "agd_ceph_writer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  class AGDCephBufferWriter : public AGDCephWriterBase {
  public:
    AGDCephBufferWriter(OpKernelConstruction *ctx) : AGDCephWriterBase(ctx) {}

  protected:
    Status SetCompressionType(OpKernelConstruction *ctx) override {
      bool compress;
      TF_RETURN_IF_ERROR(ctx->GetAttr("compress", &compress));
      header_.compression_type = compress ? format::CompressionType::GZIP : format::CompressionType::UNCOMPRESSED;
      return Status::OK();
    }

    Status WritePayload(OpKernelContext *ctx, const string &container, const string &name, const string &key, librados::bufferlist &write_buf_list) override {
      ResourceContainer<Buffer> *column;
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, name, &column));

      core::ScopedUnref column_releaser(column);
      {
        ResourceReleaser<Buffer> pool_releaser(*column);
        auto *b = column->get();
        auto &buf = *b;
        write_buf_list.push_back(ceph::buffer::create_static(buf.size(),const_cast<char*>(&buf[0])));
        TF_RETURN_IF_ERROR(SendWrite(write_buf_list, key));
        buf.reset();
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDCephBufferWriter").Device(DEVICE_CPU), AGDCephBufferWriter);
} // namespace tensorflow {
