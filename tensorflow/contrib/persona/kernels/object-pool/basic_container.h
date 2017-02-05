#pragma once

#include <memory>
#include <utility>
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
  using namespace std;

    // This is jsut a basic class to avoid having to subclass everything
  template <typename T>
  class BasicContainer : public ResourceBase {
  public:

  BasicContainer(std::unique_ptr<T> &&data) :
    data_(move(data)) {}

    T*
    get() {
      return data_.get();
    }

    string DebugString() override { return "a basic container"; }

  private:
    std::unique_ptr<T> data_;
  };
} // namespace tensorflow {
