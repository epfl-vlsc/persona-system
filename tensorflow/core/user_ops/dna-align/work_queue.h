
#ifndef TENSORFLOW_USER_OPS_WORK_QUEUE_H_
#define TENSORFLOW_USER_OPS_WORK_QUEUE_H_

#include <queue>
#include <utility>
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {


// a class wrapping STL queue 
// thread-safe, limited buffer capacity, blocks on push()
// to a full queue. 

template <typename T>
class WorkQueue {
  public:
   
    WorkQueue(int capacity);
    ~WorkQueue() {} // if you don't define the destructor, you get
                     // a weird linker error

    // return true if pushed, false otherwise
    // will block until pushed if block_ is true
    bool push(const T& item);
    // return true if success and item is valid, false otherwise
    bool pop(T& item);

    // pops everything in the queue, making size == 0
    // note that failure may 
    void pop_all(std::vector<T> &items);

    // unblock the queue, notify all threads
    void unblock();
    // set blocking behavior
    void set_block();

    bool empty() const;
    size_t capacity() const;
    size_t size() const;

  private:
    // mutex to protect the queue
    mutable mutex mu_;
    // cond vars for block/wait/notify on queue push/pop
    mutable std::condition_variable queue_pop_cv_;
    mutable std::condition_variable queue_push_cv_;
    std::queue<T> queue_;
    size_t capacity_;
    // block on calls to push, pop
    bool block_ = true;

 };

template <typename T>
void WorkQueue<T>::pop_all(std::vector<T> &items) {
  bool popped = false;
  items.clear();
  {
    mutex_lock l(mu_);
    if (queue_.empty() && block_) {
      queue_pop_cv_.wait(l, [this]() {
          return !queue_.empty() || !block_;
        });
    }

    popped = !queue_.empty();
    while (!queue_.empty()) {
      items.insert(queue_.front());
      queue_.pop();
    }
  }

  if (popped) {
    queue_push_cv_.notify_all();
  }
}

template <typename T>
bool WorkQueue<T>::pop(T& item) {

  bool popped = false;
  {
    mutex_lock l(mu_);
    //LOG_INFO << "popping work queue";
    if (queue_.empty() && block_) {
      //LOG_INFO << "pop waiting ...";
      queue_pop_cv_.wait(l, [this]() {
          return !queue_.empty() || !block_;
          });
      //LOG_INFO << "pop continuing";
    }

    if (!queue_.empty()) {
      item = queue_.front();
      queue_.pop();
      popped = true;
    }
  } 
  if (popped) {
    // tell someone blocking on write they can now write to the queue
    queue_push_cv_.notify_one();
    return true;
  } else
    return false;

}

template <typename T>
bool WorkQueue<T>::push(const T& item) {

  bool pushed = false;
  {
    mutex_lock l(mu_);
    //LOG_INFO << "pushing work queue";
    // we block until something pops and makes room for us
    // unless blocking is set to false
    if (queue_.size() == capacity_ && block_) {
      //LOG_INFO << "work queue is at capacity";
      queue_push_cv_.wait(l, [this]() {
          return (queue_.size() < capacity_) || !block_;
        });
    }

    if (queue_.size() < capacity_) {
      queue_.push(item);
      pushed = true;
    }
  }

  if (pushed) {
    // tell someone blocking on read they can now read from the queue
    queue_pop_cv_.notify_one();
    return true;
  } else
    return false;

}

template <typename T>
void WorkQueue<T>::unblock() {
  {
    mutex_lock l(mu_);
    LOG(INFO) << " unblock called!";
    block_ = false;
  }

  queue_push_cv_.notify_all();
  queue_pop_cv_.notify_all();
}

template <typename T>
void WorkQueue<T>::set_block() {
  mutex_lock l(mu_);
  block_ = true;
}

template <typename T>
WorkQueue<T>::WorkQueue(int capacity) {
  // root cause of this joke is that tensorflow attributes 
  // do not have an unsigned type. `\_(o_o)_/`
  if (capacity < 0)
    capacity_ = 10; // a sane default
  else
    capacity_ = (size_t) capacity;
}

template <typename T>
bool WorkQueue<T>::empty() const { return queue_.empty(); }

template <typename T>
size_t WorkQueue<T>::capacity() const { return capacity_; }

template <typename T>
size_t WorkQueue<T>::size() const { return queue_.size(); }

}  // namespace tensorflow

#endif

