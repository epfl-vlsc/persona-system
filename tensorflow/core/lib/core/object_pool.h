#ifndef TENSORFLOW_LIB_CORE_OBJECT_POOL_H_
#define TENSORFLOW_LIB_CORE_OBJECT_POOL_H_

#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <deque>
#include <utility>
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

template <typename T>
class ObjectPool {
public:
  typedef std::shared_ptr<T> PtrT;

  class ObjectLoan
  {
  private:
    PtrT object_;

  public:
    ObjectLoan(PtrT &object) : object_(object) {}

    T& operator*() const {
      return *object_;
    }

    T* operator->() const {
      return object_.get();
    }

    T* get() const {
      return object_.get();
    }

    void ReleaseEmpty() {
      if (get()) {
        ReturnEmpty(object_);
      }
      object_.reset();
    }

    void ReleaseReady() {
      if (get()) {
        ReturnReady(object_);
      }
      object_.reset();
    }
  };

  explicit ObjectPool(size_t max_elements, std::function<T*()> object_constructor) :
    max_elements_(max_elements), object_constructor_(object_constructor), run_(true) {}

  ~ObjectPool() {
    using namespace std;
    mutex_lock rl(ready_mu_);
    mutex_lock el(empty_mu_);
    max_elements_ = 0; // so they don't try to make any more
    empty_objects_.clear();
    ready_objects_.clear();
    run_ = false;
    ready_cv_.notify_all();
    empty_cv_.notify_all();
  }

  ObjectLoan GetReady(bool block = true) noexcept
  {
    using namespace std;
    mutex_lock l(ready_mu_);
    if (ready_objects_.empty() && block) {
      ready_cv_.wait(l, [this]() {
          ready_objects_.empty() && run_;
        });
    }

    if (!ready_objects_.empty()) {
      auto ptr = ready_objects_.pop_front();
      return ObjectLoan(ptr);
    }

    return ObjectLoan(nullptr);
  }

  ObjectLoan GetEmpty(bool block = true) noexcept
  {
    using namespace std;
    mutex_lock l(empty_mu_);
    // First, try to construct a new one if there are none available
    if (empty_objects_.empty() && all_objects_.size() < max_elements_) {
      auto raw_ptr = object_constructor_();
      if (raw_ptr != nullptr) {
        auto ptr = PtrT(raw_ptr);
        all_objects_.push_back(ptr);
        empty_objects_.push_back(ptr);
      }
    }

    // If we're still out of objects and we want to block, try to wait for one
    if (empty_objects_.empty() && block) {
      empty_cv_.wait(l, [this]() {
          empty_objects_.empty() && run_;
        });
    }

    if (!empty_objects_.empty()) {
      auto ptr = empty_objects_.pop_front();
      return ObjectLoan(ptr);
    }

    return ObjectLoan(nullptr);
  }

private:
  TF_DISALLOW_COPY_AND_ASSIGN(ObjectPool);

  void ReturnEmpty(PtrT object) noexcept
  {
    mutex_lock l(empty_mu_);
    // TODO for now, just assume that object is in all_objects_
    // because the only way to access this is via the function
    empty_objects_.push_back(object);
    empty_cv_.notify_one();
  }

  void ReturnReady(PtrT object) noexcept
  {
    mutex_lock l(ready_mu_);
    // TODO for now, just assume that object is in all_objects_
    // because the only way to access this is via the function
    ready_objects_.push_back(object);
    ready_cv_.notify_one();
  }

  mutable std::mutex ready_mu_;
  mutable std::mutex empty_mu_;
  mutable std::condition_variable ready_cv_;
  mutable std::condition_variable empty_cv_;
  std::vector<PtrT> all_objects_;
  std::deque<PtrT> ready_objects_;
  std::deque<PtrT> empty_objects_;
  std::function<T*()> object_constructor_;
  size_t max_elements_;
  volatile bool run_;
};

} // namespace tensorflow 

#endif // TENSORFLOW_LIB_CORE_OBJECT_POOL_H_
