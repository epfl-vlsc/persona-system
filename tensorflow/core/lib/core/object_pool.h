#ifndef TENSORFLOW_LIB_CORE_OBJECT_POOL_H_
#define TENSORFLOW_LIB_CORE_OBJECT_POOL_H_

#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <deque>
#include <utility>

namespace tensorflow {

template <typename T>
class ObjectPool {
 public:
  explicit ObjectPool(size_t max_elements, std::function<T*()> object_constructor);

  std::pair<std::shared_ptr<T>, std::function<void()>> GetObject(bool block = true) noexcept;

 private:
  void ReleaseObject(std::shared_ptr<T> object) noexcept;

  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  std::vector<std::shared_ptr<T>> all_objects_;
  std::deque<std::shared_ptr<T>> ready_objects_;
  std::function<T*()> object_constructor_;
  size_t max_elements_;
};

} // namespace tensorflow 

#endif // TENSORFLOW_LIB_CORE_OBJECT_POOL_H_
