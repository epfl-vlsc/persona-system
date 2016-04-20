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

namespace tensorflow {

template <typename T>
class ObjectPool {
public:
  explicit ObjectPool(size_t max_elements, std::function<T*()> object_constructor) :
  max_elements_(max_elements), object_constructor_(object_constructor) {}

  std::pair<std::shared_ptr<T>, std::function<void()>> GetObject(bool block = true) noexcept
    {
      mutex_lock l(mu_);
      if (!ready_objects_.empty()) {
        auto ptr = ready_objects_.pop_front();
        return make_pair(ptr, [this, ptr]() {
            ReleaseObject(ptr);
          });
      } else if (all_objects_.size() < max_elements_) {
        auto ptr = shared_ptr<T>(object_constructor_());
        // We can at least check if the constructor returns NULL
        // We can't verify if the memory is valid or anything
        if (ptr != nullptr) {
          all_objects_.push_back(ptr);
          return make_pair(ptr, [this, ptr]() {
              ReleaseObject(ptr);
            });
        }
      } else if (block) {
        cv_.wait(l, [this]() {
            !ready_objects_.empty();
          });
        auto ptr = ready_objects_.pop_front();
        return make_pair(ptr, [this, ptr]() {
            ReleaseObject(ptr);
          });
      }
      return make_pair(nullptr, []() {});
    }

private:
  TF_DISALLOW_COPY_AND_ASSIGN(ObjectPool);

  void ReleaseObject(std::shared_ptr<T> object) noexcept
  {
    mutex_lock l(mu_);
    // TODO for now, just assume that object is in all_objects_
    // because the only way to access this is via the function
    ready_objects_.push_back(object);
    cv_.notify_one();
  }

  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  std::vector<std::shared_ptr<T>> all_objects_;
  std::deque<std::shared_ptr<T>> ready_objects_;
  std::function<T*()> object_constructor_;
  size_t max_elements_;
};

} // namespace tensorflow 

#endif // TENSORFLOW_LIB_CORE_OBJECT_POOL_H_
