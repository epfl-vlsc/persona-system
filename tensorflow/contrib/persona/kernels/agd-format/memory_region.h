#pragma once

#include <memory>
#include <string>

#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
  class PosixMappedRegion : public ReadOnlyMemoryRegion {
  public:
    PosixMappedRegion(const void* data, uint64 size);

    // only move semantics
    PosixMappedRegion(PosixMappedRegion &&rhs);
    PosixMappedRegion& operator=(PosixMappedRegion &&rhs);

    virtual ~PosixMappedRegion() override;
    virtual const void* data() override;
    virtual uint64 length() override;

    static
    Status fromFile(const std::string &filepath, const FileSystem &fs, std::unique_ptr<ReadOnlyMemoryRegion> &result, bool synchronous = false);

  private:
    TF_DISALLOW_COPY_AND_ASSIGN(PosixMappedRegion);
    const void* data_ = nullptr;
    uint64 size_ = 0;
  };


  PosixMappedRegion fromFile(const std::string &filepath);
} // namespace tensorflow {
