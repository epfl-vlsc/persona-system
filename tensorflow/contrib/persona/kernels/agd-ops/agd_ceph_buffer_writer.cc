#include "agd_ceph_writer.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {
  using namespace std;

  class AGDCephBufferWriter : public AGDCephWriterBase {
  public:
    AGDCephBufferWriter(OpKernelConstruction *ctx) : AGDCephWriterBase(ctx) {
      bool compress;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compressed", &compress));
      header_.compression_type = compress ? format::CompressionType::GZIP : format::CompressionType::UNCOMPRESSED;
    }

  protected:

    Status WritePayload(OpKernelContext *ctx, const string &container, const string &name, const string &key, librados::bufferlist &write_buf_list) override {
      ResourceContainer<Data> *column;
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(container, name, &column));

      core::ScopedUnref column_releaser(column);
      {
        ResourceReleaser<Data> pool_releaser(*column);
        auto *b = column->get();
        write_buf_list.push_back(ceph::buffer::create_static(b->size(),const_cast<char*>(b->data())));
        TF_RETURN_IF_ERROR(SendWrite(write_buf_list, key));
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("AGDCephBufferWriter").Device(DEVICE_CPU), AGDCephBufferWriter);
} // namespace tensorflow {
