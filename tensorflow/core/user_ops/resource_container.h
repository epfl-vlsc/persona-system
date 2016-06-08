#pragma once

#include <memory>
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

template <typename T>
class ResourceContainer : public ResourceBase {
public:
  string DebugString() override;

  explicit ResourceContainer(std::unique_ptr<T> &&data) : data_(data) {}

  T* get() const {
    return data_.get();
  }

  T& operator*() const {
    return *data_;
  }

  T* operator->() const {
    return data_.get();
  }

private:
  std::unique_ptr<T> data_;
};

} // namespace tensorflow {
