#pragma once

#include <assert.h>
#include <atomic>              // NOLINT
#include <chrono>              // NOLINT
#include <condition_variable>  // NOLINT

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// helper class that allows threads to wait to be notified `min_notifies` times
class MultiNotification {
 public:
  MultiNotification() : times_notified_(0), 
    min_notifies_(1) {}

  ~MultiNotification() {
    // In case the notification is being used to synchronize its own deletion,
    // force any prior notifier to leave its critical section before the object
    // is destroyed.
    mutex_lock l(mu_);
  }

  void SetMinNotifies(int min_notifies) { min_notifies_ = min_notifies; }

  void Notify() {
    mutex_lock l(mu_);
    //assert(!HasBeenNotified());
    times_notified_++;
    
    if (times_notified_.load(std::memory_order_relaxed) >= min_notifies_)
      cv_.notify_all();
  }

  bool HasBeenNotified() const {
    return times_notified_.load(std::memory_order_relaxed) >= min_notifies_;
  }

  void WaitForNotification() {
    if (!HasBeenNotified()) {
      mutex_lock l(mu_);
      while (!HasBeenNotified()) {
        cv_.wait(l);
      }
    }
  }

 private:
  /*friend bool WaitForNotificationWithTimeout(Notification* n,
                                             int64 timeout_in_us);
  bool WaitForNotificationWithTimeout(int64 timeout_in_us) {
    bool notified = HasBeenNotified();
    if (!notified) {
      mutex_lock l(mu_);
      do {
        notified = HasBeenNotified();
      } while (!notified &&
               cv_.wait_for(l, std::chrono::microseconds(timeout_in_us)) !=
                   std::cv_status::timeout);
    }
    return notified;
  }*/

  mutex mu_;                    // protects mutations of notified_
  condition_variable cv_;       // signaled when notified_ becomes non-zero
  std::atomic<int> times_notified_;  // mutations under mu_
  int min_notifies_;
};

/*inline bool WaitForNotificationWithTimeout(Notification* n,
                                           int64 timeout_in_us) {
  return n->WaitForNotificationWithTimeout(timeout_in_us);
}*/

}  // namespace tensorflow
