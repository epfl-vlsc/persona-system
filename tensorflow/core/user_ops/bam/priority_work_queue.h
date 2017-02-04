#pragma once

#include <queue>
#include <utility>
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// a class wrapping STL queue 
// thread-safe, limited buffer capacity, blocks on push()
// to a full queue. 

template <typename T>
class PriorityWorkQueue {
  public:

    PriorityWorkQueue(int capacity);
    ~PriorityWorkQueue() { LOG(INFO) << "num pushed: " << num_push_; } // if you don't define the destructor, you get
                     // a weird linker error

    // return true if pushed, false otherwise
    // will block until pushed if block_ is true
    bool push(const T& item);
    // return true if success and item is valid, false otherwise
    bool pop(T& item);

    void drop_if_equal(T& item);

    bool peek(T& item);

    // unblock the queue, notify all threads
    void unblock();
    // set blocking behavior
    void set_block();

    bool empty() const;
    size_t capacity() const;
    size_t size() const;

    int64 num_pop_waits();
    int64 num_push_waits();
    int64 num_peek_waits();

  private:
    // mutex to protect the queue
    mutable mutex mu_;
    // cond vars for block/wait/notify on queue push/pop
    mutable std::condition_variable queue_pop_cv_;
    mutable std::condition_variable queue_push_cv_;
    std::priority_queue<T> queue_;
    size_t capacity_;
    // block on calls to push, pop
    bool block_ = true;
    int64 num_pop_waits_ = 0;
    int64 num_push_waits_ = 0;
    int64 num_peek_waits_ = 0;
    int64 num_push_ = 0;

 };

template <typename T>
bool PriorityWorkQueue<T>::peek(T& item) {
  bool popped = false;
  {
    mutex_lock l(mu_);

    if (queue_.empty() && block_) {
      num_peek_waits_++;
      queue_pop_cv_.wait(l, [this]() {
          return !queue_.empty() || !block_;
        });
    }

    if (!queue_.empty()) {
      item = queue_.top();
      popped = true;
    }
  }
  if (popped)
    queue_pop_cv_.notify_one();
  return popped;
}

template <typename T>
void PriorityWorkQueue<T>::drop_if_equal(T& item) {
  {
    mutex_lock l(mu_);
    if (!queue_.empty() && queue_.top() == item) {
      queue_.pop();
    }
  }
  queue_push_cv_.notify_one();
}

template <typename T>
bool PriorityWorkQueue<T>::pop(T& item) {

  bool popped = false;
  {
    mutex_lock l(mu_);
    if (queue_.empty() && block_) {
      num_pop_waits_++;
      queue_pop_cv_.wait(l, [this]() {
          return !queue_.empty() || !block_;
          });
    }

    if (!queue_.empty()) {
      item = queue_.top();
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
bool PriorityWorkQueue<T>::push(const T& item) {

  bool pushed = false;
  {
    mutex_lock l(mu_);
    // we block until something pops and makes room for us
    // unless blocking is set to false
    if (queue_.size() == capacity_ && block_) {
      num_push_waits_++;
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
    // TODO maybe notify_all is better here? If so, good to drop the notify_one in peek
    queue_pop_cv_.notify_one();
    num_push_++;
    return true;
  } else
    return false;

}

template <typename T>
void PriorityWorkQueue<T>::unblock() {
  {
    mutex_lock l(mu_);
    VLOG(INFO) << "PriorityWorkQueue("<< this << ") unblock called!";
    block_ = false;
  }

  queue_push_cv_.notify_all();
  queue_pop_cv_.notify_all();
}

template <typename T>
void PriorityWorkQueue<T>::set_block() {
  mutex_lock l(mu_);
  block_ = true;
}

template <typename T>
PriorityWorkQueue<T>::PriorityWorkQueue(int capacity) {
  // root cause of this joke is that tensorflow attributes 
  // do not have an unsigned type. `\_(o_o)_/`
  if (capacity < 0)
    capacity_ = 10; // a sane default
  else
    capacity_ = (size_t) capacity;
}

template <typename T>
bool PriorityWorkQueue<T>::empty() const { return queue_.empty(); }

template <typename T>
size_t PriorityWorkQueue<T>::capacity() const { return capacity_; }

template <typename T>
size_t PriorityWorkQueue<T>::size() const { return queue_.size(); }

template <typename T>
int64 PriorityWorkQueue<T>::num_pop_waits() { return num_pop_waits_; }

template <typename T>
int64 PriorityWorkQueue<T>::num_push_waits() { return num_push_waits_; }

template <typename T>
int64 PriorityWorkQueue<T>::num_peek_waits() { return num_peek_waits_; }

}  // namespace tensorflow

