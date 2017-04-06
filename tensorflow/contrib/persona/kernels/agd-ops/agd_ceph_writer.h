#pragma once
#include "agd_writer.h"
#include <rados/librados.hpp>
#include <rados/buffer.h>

namespace tensorflow {
  class AGDCephWriterBase : public AGDWriterBase {
  public:
    AGDCephWriterBase(OpKernelConstruction *ctx);
    void Compute(OpKernelContext* ctx) override final;

  protected:
    virtual Status WritePayload(OpKernelContext *ctx, const std::string &container, const std::string &name, const std::string &key, librados::bufferlist &write_buf_list) = 0;
    Status SendWrite(librados::bufferlist &write_buf_list, const std::string &key);

  private:
    Status SetPool(OpKernelContext *ctx);
    std::string cluster_name_, user_name_, pool_name_, ceph_conf_path_;
    librados::Rados cluster_;
    librados::IoCtx io_ctx_;
    librados::bufferlist write_buf_list_;
  };
} // namespace tensorflow {
