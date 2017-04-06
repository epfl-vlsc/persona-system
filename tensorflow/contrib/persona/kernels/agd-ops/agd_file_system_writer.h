#pragma once

#include "agd_writer.h"
namespace tensorflow {
  class AGDFileSystemWriterBase : public AGDWriterBase {
  public:
    AGDFileSystemWriterBase(OpKernelConstruction *ctx);
    void Compute(OpKernelContext* ctx) override final;
  protected:

    virtual Status WriteResource(OpKernelContext *ctx, FILE *f, const std::string &container,
                                 const std::string &name) = 0;

    Status WriteData(FILE *f, const char* data, size_t size);
  };
} // namespace tensorflow {
