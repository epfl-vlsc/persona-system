#pragma once

#include "tensorflow/contrib/persona/kernels/lockfree/concurrentqueue.h"
#include "tensorflow/core/lib/core/status.h"
#include <functional>

namespace tensorflow {
  template <typename T>
  class TaskRunner {
  public:

    TaskRunner(Env *env, size_t num_threads, const std::string &name = "TaskRunner") {
      workers_.reset(new thread::ThreadPool(env, name, num_threads));
    }

    bool Enqueue(T &t) final {
      return queue_.enqueue(t);
    }

    bool EnqueueMany(T *t, size_t count) final {
      return queue_.enqueue_bulk(t, count);
    }

    Status Start() {
      return Status::OK();
    };

  protected:
    bool DequeueOne(T &t) final {
      return queue_.try_dequeue(t);
    };

    std::function<void()> RunWorker(std::function< std::function< bool(T&) >> worker) {
      using namespace std;
      auto wrapper = [this, worker]() {
        auto before = num_workers_.fetch_add(1, memory_order_relaxed);
        if (before == 0) {
          active_ = true;
        }
        worker(this->Dequeue);
        before = num_workers_.fetch_sub(1, memory_order_relaxed);
        if (before == 1) {
          active_ = false;
        }
      };

      return wrapper;
    }

    void AddWorker(std::function<void()> worker) {
      using namespace std;
      workers_->Schedule(worker);
      auto before = num_workers_.fetch_add(1, memory_order_relaxed);
      if (before == 0) {
        active_ = true;
      }
    }


  private:
    moodycamel::ConcurrentQueue<T> queue_;
    std::unique_ptr<thread::ThreadPool> workers_;
    std::atomic_uint_fast16_t num_workers_{0};
    volatile bool active_ = false;
  };
} // namespace tensorflow {
