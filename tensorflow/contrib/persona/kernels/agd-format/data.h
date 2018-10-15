#pragma once

#include <cstddef>

namespace tensorflow {
  class Data {
  public:
    virtual ~Data() = default;
    virtual const char* data() const = 0;
    virtual char* mutable_data();
    virtual std::size_t size() const = 0;
    virtual void release();
  };

  class DataReleaser {
  public:
    DataReleaser(Data &data);
    ~DataReleaser();
  private:
    Data &data_;
  };
} // namespace tensorflow {
