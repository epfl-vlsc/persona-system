/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/lib/core/threadpool.h"

#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace thread {

struct EigenEnvironment {
  typedef Thread EnvThread;
  struct Task {
    std::function<void()> f;
    uint64 trace_id;
  };

  Env* const env_;
  const ThreadOptions thread_options_;
  const string name_;
  int current_cpu_;

  EigenEnvironment(Env* env, const ThreadOptions& thread_options,
                   const string& name, int affinity_start)
      : env_(env), thread_options_(thread_options), name_(name),
        current_cpu_(affinity_start){}

  EnvThread* CreateThread(std::function<void()> f) {
    Thread* t = env_->StartThread(thread_options_, name_, [=]() {
      // Set the processor flag to flush denormals to zero
      port::ScopedFlushDenormal flush;
      f();
    });
    if (current_cpu_ >= port::NumSchedulableCPUs()) {
      current_cpu_ = 0;
      /*LOG(INFO) << "uh oh, trying to set affinity on core " << 
        current_cpu_ << ", which is greater than " << port::NumSchedulableCPUs();*/
    }
    //LOG(INFO) << "Setting thread affinity to core: " << current_cpu_;
    Status status = Status::OK();
    if (current_cpu_ != -1) {
      status = t->SetAffinity(current_cpu_);
      current_cpu_++;
    }
    if (!status.ok()) {
      LOG(INFO) << "Set affinity failed in impl create thread";
    }
    return t;
  }

  Task CreateTask(std::function<void()> f) {
    uint64 id = 0;
    if (port::Tracing::IsActive()) {
      id = port::Tracing::UniqueId();
      port::Tracing::RecordEvent(port::Tracing::EventCategory::kScheduleClosure,
                                 id);
    }
    return Task{std::move(f), id};
  }

  void ExecuteTask(const Task& t) {
    if (t.trace_id != 0) {
      port::Tracing::ScopedActivity region(
          port::Tracing::EventCategory::kRunClosure, t.trace_id);
      t.f();
    } else {
      t.f();
    }
  }
};

struct ThreadPool::Impl : Eigen::ThreadPoolTempl<EigenEnvironment> {
  Impl(Env* env, const ThreadOptions& thread_options, const string& name,
       int num_threads, int affinity_start)
      : Eigen::ThreadPoolTempl<EigenEnvironment>(
            num_threads, EigenEnvironment(env, thread_options, name, affinity_start)),
        num_threads_(num_threads) {}

  void ParallelFor(int64 total, int64 cost_per_unit,
                   std::function<void(int64, int64)> fn) {
#ifdef EIGEN_USE_NONBLOCKING_THREAD_POOL
    CHECK_GE(total, 0);
    CHECK_EQ(total, (int64)(Eigen::Index)total);
    Eigen::ThreadPoolDevice device(this, num_threads_);
    device.parallelFor(
        total, Eigen::TensorOpCost(0, 0, cost_per_unit),
        [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
#else
    CHECK(0);  // should not be used with the old thread pool
#endif
  }

  int NumThreads() const { return num_threads_; };

  const int num_threads_;
};

  ThreadPool::ThreadPool(Env* env, const string& name, int num_threads, int affinity_start)
    : ThreadPool(env, ThreadOptions(), name, num_threads, affinity_start) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads, int affinity_start) {
  CHECK_GE(num_threads, 1);
  impl_.reset(
      new ThreadPool::Impl(env, thread_options, "tf_" + name, num_threads, affinity_start));
}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  impl_->Schedule(std::move(fn));
}

void ThreadPool::ParallelFor(int64 total, int64 cost_per_unit,
                             std::function<void(int64, int64)> fn) {
  impl_->ParallelFor(total, cost_per_unit, std::move(fn));
}

int ThreadPool::NumThreads() const { return impl_->NumThreads(); }

}  // namespace thread
}  // namespace tensorflow
