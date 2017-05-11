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
    librados::Rados cluster_;
    librados::IoCtx io_ctx_;
    librados::bufferlist write_buf_list_;
  };
} // namespace tensorflow {
