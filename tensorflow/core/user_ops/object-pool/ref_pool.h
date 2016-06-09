#pragma once

#include "tensorflow/core/framework/resource_mgr.h"
#include <string>
#include <deque>
#include <condition_variable>
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

  class ReferencePool : public ResourceBase {
  public:
    virtual ~ReferencePool() override;
    // For initialization, locks are not grabbed here
    void AddResource(const std::string &container, const std::string &name);

    // For getting / releasing resources
    // locks are grabbed here
    void ReturnResource(const std::string &container, const std::string &name);
    void GetResource(std::string &container, std::string &name);

    string DebugString() override;
  private:
    std::deque<std::pair<std::string, std::string>> objects_;
    mutable mutex objects_mu_;
    mutable std::condition_variable objects_cv_;
    volatile bool run_ = true;
  };

} // namespace tensorflow {
