#pragma once

#include "tensorflow/core/framework/resource_mgr.h"
#include <string>
#include <deque>
#include <memory>
#include <condition_variable>
#include "tensorflow/core/platform/mutex.h"
#include "resource_container.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

  template <typename U>
  class ResourceContainer;

  template <typename T>
  class ReferencePool : public ResourceBase {
  public:

    virtual ~ReferencePool() override {
      mutex_lock l(objects_mu_);
      objects_.clear();
      run_ = false;
      objects_cv_.notify_all();
    }

    // For initialization, locks are not grabbed here
    void AddResource(std::unique_ptr<ResourceContainer<T>> &&new_resource)
    {
      objects_.push_back(new_resource.get());
      all_objects_.push_back(new_resource.release());
    }

    // For getting / releasing resources
    // locks are grabbed here
    Status GetResource(ResourceContainer<T> **rct) {
      mutex_lock l(objects_mu_);
      // while instead of for in case multiple are woken up
      while (objects_.empty() && run_) {
        objects_cv_.wait(l, [this]() -> bool {
            return !objects_.empty() && run_;
          });
      }

      if (run_) {
        // guaranteed that there is at least one item in the queue now
        auto s = objects_.front();
        *rct = s;
        objects_.pop_front();
        return Status::OK();
      } else {
        return errors::Aborted("Reference Pool has been stopped");
      }
    }

    string DebugString() override {
      static const string s = "a reference pool";
      return s;
    }

  private:

    std::vector<ResourceContainer<T>*> all_objects_;
    std::deque<ResourceContainer<T>*> objects_;
    mutable mutex objects_mu_;
    mutable std::condition_variable objects_cv_;
    volatile bool run_ = true;

    void ReleaseResource(ResourceContainer<T> *rct) {
      if (run_) {
        {
          mutex_lock l(objects_mu_);
          objects_.push_front(rct);
        }
        objects_cv_.notify_one();
      }
    }

    friend class ResourceContainer<T>;
  };

} // namespace tensorflow {
