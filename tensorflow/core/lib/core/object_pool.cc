#include "object_pool.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

using namespace std;

template <typename T>
ObjectPool<T>::ObjectPool(size_t max_elements, std::function<T*()> object_constructor) :
  max_elements_(max_elements), object_constructor_(object_constructor)
{
}

template <typename T>
pair<shared_ptr<T>, function<void()>> ObjectPool<T>::GetObject(bool block) noexcept
{
  mutex_lock l(mu_);
  if (!ready_objects_.empty()) {
    auto ptr = ready_objects_.pop_front();
    return make_pair(ptr, [this, ptr]() {
        ReleaseObject(ptr);
      });
  } else if (all_objects_.size() < max_elements_) {
    auto ptr = shared_ptr<T>(object_constructor_());
    // TODO what if object_constructor_ returns something null?
    all_objects_.push_back(ptr);
    return make_pair(ptr, [this, ptr]() {
        ReleaseObject(ptr);
      });
  } else if (block) {
    cv_.wait(l, [this]() {
        !ready_objects_.empty();
      });
    auto ptr = ready_objects_.pop_front();
    return make_pair(ptr, [this, ptr]() {
        ReleaseObject(ptr);
      });
  } else {
    return make_pair(nullptr, []() {});
  }
}

template <typename T>
void ObjectPool<T>::ReleaseObject(std::shared_ptr<T> object) noexcept
{
  mutex_lock l(mu_);
  // TODO for now, just assume that object is in all_objects_
  // because the only way to access this is via the function
  ready_objects_.push_back(object);
  cv_.notify_one();
}

}// namespace tensorflow 
