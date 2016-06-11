#pragma once

#include <memory>
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

template <typename T>
class ResourceContainer : public ResourceBase {
public:
  string DebugString() override {
    static const string s = "a resource container";
    return s;
  };

  explicit ResourceContainer(std::unique_ptr<T> &&data) : data_(std::move(data)) {}

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
