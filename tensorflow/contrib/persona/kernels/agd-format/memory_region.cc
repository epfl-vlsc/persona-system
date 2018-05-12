#include "memory_region.h"

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "tensorflow/core/lib/core/errors.h"


namespace tensorflow {

  using namespace std;
  using namespace errors;

  PosixMappedRegion::PosixMappedRegion(const void* data, uint64 size) :
    data_(data), size_(size) {}

  PosixMappedRegion::~PosixMappedRegion() {
    // TODO should log if this is non-0
    if (data_) {
      munmap(const_cast<void*>(data_), size_);
    }
  }

  const void* PosixMappedRegion::data() {
    return data_;
  }

  uint64 PosixMappedRegion::length() {
    return size_;
  }

  PosixMappedRegion::PosixMappedRegion(PosixMappedRegion &&rhs) {
    data_ = rhs.data_; rhs.data_ = nullptr;
    size_ = rhs.size_; rhs.size_ = 0;
  }

  PosixMappedRegion& PosixMappedRegion::operator=(PosixMappedRegion &&rhs) {
    data_ = rhs.data_; rhs.data_ = nullptr;
    size_ = rhs.size_; rhs.size_ = 0;
    return *this;
  }

  Status PosixMappedRegion::fromFile(const string &filepath, const FileSystem &fs,
                                     unique_ptr<ReadOnlyMemoryRegion> &result, bool synchronous) {
    string translated_fname = fs.TranslateName(filepath);
    Status s = Status::OK();
    int fd = open(translated_fname.c_str(), O_RDONLY);
    if (fd < 0) {
      s = Internal("PosixMappedRegion: file open ", filepath, " (translated: ", translated_fname, ") failed with errno ", errno);
    } else {
      struct stat st;
      ::fstat(fd, &st);
      const void* address =
#ifndef __APPLE__
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE | (synchronous ? MAP_POPULATE : 0), fd, 0);
#else
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); // appuru not have map_populate
#endif
      if (address == MAP_FAILED) {
        s = Internal("PosixMappedRegion: mmap ", filepath, " (translated: ", translated_fname, ") failed with errno ", errno);
      } else {
        result.reset(new PosixMappedRegion(address, st.st_size));
      }
      close(fd);
    }
    return s;
  }
} // namespace tensorflow {
