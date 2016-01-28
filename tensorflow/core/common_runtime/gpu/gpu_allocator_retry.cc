/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_allocator_retry.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

GPUAllocatorRetry::GPUAllocatorRetry() : env_(Env::Default()) {}

void* GPUAllocatorRetry::AllocateRaw(
    std::function<void*(size_t alignment, size_t num_bytes,
                        bool verbose_failure)> alloc_func,
    int max_millis_to_wait, size_t alignment, size_t num_bytes) {
  if (num_bytes == 0) {
    LOG(WARNING) << "Request to allocate 0 bytes";
    return nullptr;
  }
  uint64 deadline_micros = env_->NowMicros() + max_millis_to_wait * 1000;
  void* ptr = nullptr;
  while (ptr == nullptr) {
    ptr = alloc_func(alignment, num_bytes, false);
    if (ptr == nullptr) {
      uint64 now = env_->NowMicros();
      if (now < deadline_micros) {
        mutex_lock l(mu_);
        WaitForMilliseconds(&l, &memory_returned_,
                            (deadline_micros - now) / 1000);
      } else {
        return alloc_func(alignment, num_bytes, true);
      }
    }
  }
  return ptr;
}

void GPUAllocatorRetry::DeallocateRaw(std::function<void(void*)> dealloc_func,
                                      void* ptr) {
  if (ptr == nullptr) {
    LOG(ERROR) << "Request to free nullptr";
    return;
  }
  dealloc_func(ptr);
  {
    mutex_lock l(mu_);
    memory_returned_.notify_all();
  }
}

}  // namespace tensorflow
