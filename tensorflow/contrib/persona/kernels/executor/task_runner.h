#pragma once

#include "tensorflow/contrib/persona/kernels/lockfree/concurrentqueue.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include <functional>
#include <thread>

namespace tensorflow {
  template <typename T>
  class TaskRunner {
  public:

    typedef std::function<void (std::function<bool (T&)>) > ThreadFunction;

    TaskRunner(Env *env, size_t num_threads, const std::string &name = "TaskRunner", uint_fast64_t polling_sleep_microseconds = 50) :
            poll_interval_(polling_sleep_microseconds) {
      workers_.reset(new thread::ThreadPool(env, name, num_threads));
    }

    ~TaskRunner() {
      Stop();
    }

    bool Enqueue(T &t) {
      return queue_.enqueue(t);
    }

    bool EnqueueMany(T *t, size_t count) {
      return queue_.enqueue_bulk(t, count);
    }

    virtual Status Start() = 0;

    void Stop() {
      using namespace std;
      active_ = false;
      while (num_workers_.load(memory_order_relaxed) != 0) {
        this_thread::sleep_for(shutdown_pause_);
      }
    }

  protected:
    bool DequeueOne(T &t) {
      while (active_) {
        if (queue_.try_dequeue(t)) {
          return true;
        }
        std::this_thread::sleep_for(poll_interval_);
      }
      return false;
    };

    std::function<void()> RunWorker(ThreadFunction worker) {
      using namespace std;
      auto wrapper = [this, worker]() {
        auto before = num_workers_.fetch_add(1, memory_order_relaxed);
        if (before == 0) {
          active_ = true;
        }
        LOG(INFO) << "Starting working " << before;
        worker([this](T &t) { return DequeueOne(t); });
        before = num_workers_.fetch_sub(1, memory_order_relaxed);
        if (before == 1) {
          active_ = false;
        }
        LOG(INFO) << "Stopping working " << before-1;
      };

      return wrapper;
    }

    void AddWorker(ThreadFunction worker, uint_fast16_t worker_count = 1) {
      using namespace std;
      for (uint_fast16_t i = 0; i < worker_count; ++i) {
        // TODO not sure if we need to call Runworker each time
        // try without it to see if that works?
        workers_->Schedule(RunWorker(worker));
      }
    }

  private:
    moodycamel::ConcurrentQueue<T> queue_;
    std::unique_ptr<thread::ThreadPool> workers_;
    std::atomic_uint_fast16_t num_workers_{0};
    volatile bool active_ = false;
    std::chrono::milliseconds shutdown_pause_{10};
    std::chrono::microseconds poll_interval_;
  };
} // namespace tensorflow {
