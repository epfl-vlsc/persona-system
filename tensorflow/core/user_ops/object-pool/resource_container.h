#pragma once

#include <memory>
#include <string>
#include "tensorflow/core/framework/resource_mgr.h"
#include "ref_pool.h"

namespace tensorflow {

  template <typename U>
  class ReferencePool;

template <typename T>
class ResourceContainer : public ResourceBase {
public:
  string DebugString() override {
    static const string s = "a resource container";
    return s;
  };

  explicit ResourceContainer(std::unique_ptr<T> &&data, const std::string &container, const std::string &name, ReferencePool<T> *rp) : data_(std::move(data)), container_(container), name_(name), ref_pool_(rp) {}

  T* get() const {
    return data_.get();
  }

  void release() {
    ref_pool_->ReleaseResource(this);
  }

  const std::string& container() { return container_; }
  const std::string& name() { return name_; }

private:
  std::unique_ptr<T> data_;
  ReferencePool<T> *ref_pool_;
  std::string container_, name_;
 };

template <typename T>
class ResourceReleaser {
public:
  explicit ResourceReleaser(ResourceContainer<T> &rc) : rc_(rc) {}
  ~ResourceReleaser() {
    rc_.release();
  }

private:
  ResourceContainer<T> &rc_;
};

} // namespace tensorflow {
