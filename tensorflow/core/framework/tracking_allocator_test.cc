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

#include "tensorflow/core/framework/tracking_allocator.h"

#include <unordered_map>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class TestableSizeTrackingAllocator : public Allocator {
 public:
  string Name() override { return "test"; }
  void* AllocateRaw(size_t /*alignment*/, size_t num_bytes) override {
    void* ptr = malloc(num_bytes);
    size_map_[ptr] = num_bytes;
    return ptr;
  }
  void DeallocateRaw(void* ptr) override {
    const auto& iter = size_map_.find(ptr);
    EXPECT_NE(size_map_.end(), iter);
    size_map_.erase(iter);
    free(ptr);
  }
  bool TracksAllocationSizes() override { return true; }
  size_t RequestedSize(void* ptr) override {
    const auto& iter = size_map_.find(ptr);
    EXPECT_NE(size_map_.end(), iter);
    return iter->second;
  }

 private:
  std::unordered_map<void*, size_t> size_map_;
};

class NoMemoryAllocator : public Allocator {
 public:
  string Name() override { return "test"; }
  void* AllocateRaw(size_t /*alignment*/, size_t num_bytes) override {
    return nullptr;
  }
  void DeallocateRaw(void* ptr) override {}
  bool TracksAllocationSizes() override { return true; }
};

TEST(TrackingAllocatorTest, SimpleNoTracking) {
  Allocator* a = cpu_allocator();

  EXPECT_FALSE(a->TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(a);

  void* p1 = ta->AllocateRaw(4, 4);
  ta->DeallocateRaw(p1);
  void* p2 = ta->AllocateRaw(4, 12);

  std::pair<size_t, size_t> sizes = ta->GetSizesAndUnRef();

  EXPECT_EQ(16, sizes.first);
  EXPECT_EQ(0, sizes.second);

  ta->DeallocateRaw(p2);
}

TEST(TrackingAllocatorTest, SimpleTracking) {
  TestableSizeTrackingAllocator a = TestableSizeTrackingAllocator();

  EXPECT_TRUE(a.TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(&a);

  void* p1 = ta->AllocateRaw(4, 12);
  ta->DeallocateRaw(p1);
  void* p2 = ta->AllocateRaw(4, 4);

  std::pair<size_t, size_t> sizes = ta->GetSizesAndUnRef();

  EXPECT_EQ(16, sizes.first);
  EXPECT_EQ(12, sizes.second);

  ta->DeallocateRaw(p2);
}

TEST(TrackingAllocatorTest, OutOfMemory) {
  NoMemoryAllocator a;

  EXPECT_TRUE(a.TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(&a);

  void* p1 = ta->AllocateRaw(4, 12);
  EXPECT_EQ(nullptr, p1);

  std::pair<size_t, size_t> sizes = ta->GetSizesAndUnRef();

  EXPECT_EQ(0, sizes.first);
  EXPECT_EQ(0, sizes.second);
}

TEST(TrackingAllocatorTest, FreeNullPtr) {
  NoMemoryAllocator a;

  EXPECT_TRUE(a.TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(&a);

  ta->DeallocateRaw(nullptr);

  std::pair<size_t, size_t> sizes = ta->GetSizesAndUnRef();

  EXPECT_EQ(0, sizes.first);
  EXPECT_EQ(0, sizes.second);
}

}  // namespace tensorflow
