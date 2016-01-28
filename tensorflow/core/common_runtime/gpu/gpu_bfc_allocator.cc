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

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_allocator_retry.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {

GPUBFCAllocator::GPUBFCAllocator(int device_id, size_t total_memory)
    : device_id_(device_id), next_allocation_id_(1) {
  // Get a pointer to the stream_executor for this device
  stream_exec_ = GPUMachineManager()->ExecutorForDevice(device_id).ValueOrDie();

  // Allocate the requested amount of memory.
  gpu_memory_size_ = total_memory;

  // Create a bunch of bins of various good sizes.

  // Covers allocations of exactly 256 bytes (the minimum size).
  bins_.insert(std::make_pair(256, new Bin(256)));

  // We create bins to fit all possible ranges that cover the
  // gpu_memory_size_ starting from allocations up to 1024 bytes to
  // allocations up to (and including) the memory limit.
  for (size_t bin_size = 1024; bin_size < gpu_memory_size_ * 2; bin_size *= 2) {
    LOG(INFO) << "Creating bin of max chunk size "
              << strings::HumanReadableNumBytes(bin_size);
    bins_.insert(std::make_pair(bin_size, new Bin(bin_size)));
  }
}

GPUBFCAllocator::~GPUBFCAllocator() {
  // Return memory back.
  if (base_ptr_) {
    gpu::DeviceMemoryBase gpu_ptr{base_ptr_};
    stream_exec_->Deallocate(&gpu_ptr);
  }

  gtl::STLDeleteValues(&bins_);
  gtl::STLDeleteValues(&ptr_to_chunk_map_);
}

void GPUBFCAllocator::MaybeInitialize() {
  if (base_ptr_ != nullptr) {
    return;
  }

  LOG(INFO) << "Allocating " << strings::HumanReadableNumBytes(gpu_memory_size_)
            << " bytes.";
  gpu::DeviceMemory<char> gpu_mem =
      stream_exec_->AllocateArray<char>(gpu_memory_size_);

  QCHECK(gpu_mem != nullptr)
      << " Could not allocate GPU device memory for device " << device_id_
      << ". Tried to allocate "
      << strings::HumanReadableNumBytes(gpu_memory_size_);
  base_ptr_ = gpu_mem.opaque();
  LOG(INFO) << "GPU " << device_id_ << " memory begins at " << base_ptr_
            << " extends to "
            << static_cast<void*>(
                   (static_cast<char*>(base_ptr_) + gpu_memory_size_));

  // Create one large chunk for the whole memory space that will
  // be chunked later.
  GPUBFCAllocator::Chunk* c = new GPUBFCAllocator::Chunk();
  c->ptr = gpu_mem.opaque();
  c->size = gpu_memory_size_;
  c->allocation_id = -1;
  c->prev = nullptr;
  c->next = nullptr;

  ptr_to_chunk_map_.insert(std::make_pair(c->ptr, c));

  // Insert the chunk into the right bin.
  InsertFreeChunkIntoBin(c);

  // Invoke visitors on newly allocated region.
  for (auto visitor : region_visitors_) {
    visitor(base_ptr_, gpu_memory_size_);
  }
}

void* GPUBFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes) {
  static const int64 kMaxMillisToWait = 10000;  // 10 seconds
  return retry_helper_.AllocateRaw(
      [this](size_t a, size_t nb, bool v) {
        return AllocateRawInternal(a, nb, v);
      },
      kMaxMillisToWait, unused_alignment, num_bytes);
}

void* GPUBFCAllocator::AllocateRaw(
    size_t unused_alignment, size_t num_bytes,
    const AllocationAttributes& allocation_attr) {
  if (allocation_attr.no_retry_on_failure) {
    // Return immediately upon the first failure if this is for allocating an
    // optional scratch space.
    void* result = AllocateRawInternal(unused_alignment, num_bytes, false);
    if (result == nullptr) {
      // The counter incrementing is not thread-safe. But we don't really care.
      // TODO(zhengxq): we should implement a LOG_FIRST_N and LOG_EVERY_N for
      // more general usage.
      static int log_counter = 0;
      if (log_counter < 10) {
        log_counter++;
        LOG(WARNING)
            << "Ran out of memory trying to allocate "
            << strings::HumanReadableNumBytes(num_bytes)
            << ". The caller indicates that this is not a failure, but"
            << " may mean that there could be performance gains if more"
            << " memory is available.";
      }
    }
    return result;
  } else {
    return AllocateRaw(unused_alignment, num_bytes);
  }
}

void* GPUBFCAllocator::AllocateRawInternal(size_t unused_alignment,
                                           size_t num_bytes,
                                           bool dump_log_on_failure) {
  if (num_bytes == 0) {
    LOG(ERROR) << "tried to allocate 0 bytes";
    return nullptr;
  }
  // First, always allocate memory of at least 256 bytes, and always
  // allocate multiples of 256 bytes so all memory addresses are
  // nicely byte aligned.
  size_t rounded_bytes = (256 * ((num_bytes + 255) / 256));
  DCHECK_EQ(0, rounded_bytes % 256);

  // The BFC allocator tries to find the best fit first.
  //
  // First identify the first bin that could satisfy rounded_bytes.
  auto it = bins_.lower_bound(rounded_bytes);
  if (it == bins_.end()) {
    LOG(ERROR) << " Asked for " << rounded_bytes << " but largest bin was "
               << bins_.rbegin()->first;
    return nullptr;
  }

  mutex_lock l(lock_);
  MaybeInitialize();

  for (; it != bins_.end(); ++it) {
    // Start searching from the first bin for the smallest chunk that fits
    // rounded_bytes.
    Bin* b = it->second;
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
         ++citer) {
      GPUBFCAllocator::Chunk* chunk = (*citer);
      DCHECK(!chunk->in_use());
      if (chunk->size >= rounded_bytes) {
        // We found an existing chunk that fits us that wasn't in use, so remove
        // it from the free bin structure prior to using.
        RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

        // If we can break the size of the chunk into two reasonably
        // large pieces, do so.
        //
        // TODO(vrv): What should be the criteria when deciding when
        // to split?
        if (chunk->size >= rounded_bytes * 2) {
          SplitChunk(chunk, rounded_bytes);
        }

        // The requested size of the returned chunk is what the user
        // has allocated.
        chunk->requested_size = num_bytes;
        // Assign a unique id and increment the id counter, marking the
        // chunk as being in use.
        chunk->allocation_id = next_allocation_id_++;

        VLOG(4) << "Returning: " << chunk->ptr;
        return chunk->ptr;
      }
    }
  }

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  // Dump the memory log for analysis.
  if (dump_log_on_failure) {
    DumpMemoryLog(rounded_bytes);
    LOG(WARNING) << "Ran out of memory trying to allocate "
                 << strings::HumanReadableNumBytes(num_bytes)
                 << ".  See logs for memory state";
  }
  return nullptr;
}

void GPUBFCAllocator::SplitChunk(GPUBFCAllocator::Chunk* c, size_t num_bytes) {
  CHECK(!c->in_use() && !c->bin);

  // Create a new chunk starting num_bytes after c
  GPUBFCAllocator::Chunk* new_chunk = new GPUBFCAllocator::Chunk();
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  VLOG(6) << "Adding to chunk map: " << new_chunk->ptr;
  ptr_to_chunk_map_.insert(std::make_pair(new_chunk->ptr, new_chunk));

  // Set the new sizes of the chunks.
  new_chunk->size = c->size - num_bytes;
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->allocation_id = -1;

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  GPUBFCAllocator::Chunk* c_neighbor = c->next;
  new_chunk->prev = c;
  new_chunk->next = c_neighbor;
  c->next = new_chunk;
  if (c_neighbor) {
    c_neighbor->prev = new_chunk;
  }

  // Add the newly free chunk to the free bin.
  InsertFreeChunkIntoBin(new_chunk);
}

void GPUBFCAllocator::DeallocateRaw(void* ptr) {
  retry_helper_.DeallocateRaw([this](void* p) { DeallocateRawInternal(p); },
                              ptr);
}

void GPUBFCAllocator::DeallocateRawInternal(void* ptr) {
  if (ptr == nullptr) {
    LOG(ERROR) << "tried to deallocate nullptr";
    return;
  }
  mutex_lock l(lock_);

  // Find the chunk from the ptr.
  auto it = ptr_to_chunk_map_.find(ptr);
  CHECK(it != ptr_to_chunk_map_.end())
      << "Asked to deallocate a pointer we never allocated: " << ptr;

  GPUBFCAllocator::Chunk* c = it->second;
  VLOG(6) << "Chunk at " << c->ptr << " no longer in use";

  // Consider coalescing it.
  FreeAndMaybeCoalesce(c);
}

// Merges c1 and c2 when c1->next is c2 and c2->prev is c1.
// We merge c2 into c1.
void GPUBFCAllocator::Merge(GPUBFCAllocator::Chunk* c1,
                            GPUBFCAllocator::Chunk* c2) {
  // We can only merge chunks that are not in use.
  CHECK(!c1->in_use() && !c2->in_use());

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3
  GPUBFCAllocator::Chunk* c3 = c2->next;
  c1->next = c3;
  CHECK(c2->prev == c1);
  if (c3 != nullptr) {
    c3->prev = c1;
  }

  // Set the new size
  c1->size += c2->size;

  DeleteChunk(c2);
}

void GPUBFCAllocator::DeleteChunk(Chunk* c) {
  // Delete c2 and cleanup all state
  VLOG(4) << "Removing: " << c->ptr;
  ptr_to_chunk_map_.erase(c->ptr);
  delete c;
}

void GPUBFCAllocator::InsertFreeChunkIntoBin(GPUBFCAllocator::Chunk* c) {
  CHECK(!c->in_use() && !c->bin);
  auto it = bins_.lower_bound(c->size);
  CHECK(it != bins_.end()) << " Tried to reassign to non-existent bin for size "
                           << c->size;
  Bin* new_bin = it->second;
  c->bin = new_bin;
  new_bin->free_chunks.insert(c);
}

void GPUBFCAllocator::RemoveFreeChunkIterFromBin(
    GPUBFCAllocator::Bin::FreeChunkSet* free_chunks,
    const GPUBFCAllocator::Bin::FreeChunkSet::iterator& citer) {
  GPUBFCAllocator::Chunk* c = *citer;
  CHECK(!c->in_use() && c->bin);
  free_chunks->erase(citer);
  c->bin = nullptr;
}

void GPUBFCAllocator::RemoveFreeChunkFromBin(GPUBFCAllocator::Chunk* c) {
  CHECK(!c->in_use() && c->bin);
  int count = c->bin->free_chunks.erase(c);
  CHECK(count > 0) << "Could not find chunk in bin";
  c->bin = nullptr;
}

void GPUBFCAllocator::FreeAndMaybeCoalesce(GPUBFCAllocator::Chunk* c) {
  CHECK(c->in_use() && !c->bin);

  // Mark the chunk as no longer in use
  c->allocation_id = -1;

  // This chunk is no longer in-use, consider coalescing the chunk
  // with adjacent chunks.
  Chunk* chunk_to_reassign = c;

  // If the next chunk is free, coalesce the two, if the result would
  // fit in an existing bin.
  if (c->next && !c->next->in_use()) {
    VLOG(8) << "Chunk at " << c->next->ptr << " merging with c " << c->ptr;

    chunk_to_reassign = c;

    // Deletes c->next
    RemoveFreeChunkFromBin(c->next);
    Merge(c, c->next);
  }

  // If the previous chunk is free, coalesce the two
  if (c->prev && !c->prev->in_use()) {
    VLOG(8) << "Chunk at " << c->ptr << " merging into c->prev "
            << c->prev->ptr;

    chunk_to_reassign = c->prev;

    // Deletes c
    RemoveFreeChunkFromBin(c->prev);
    Merge(c->prev, c);
  }

  InsertFreeChunkIntoBin(chunk_to_reassign);
}

void GPUBFCAllocator::AddAllocVisitor(Visitor visitor) {
  VLOG(1) << "AddVisitor";
  mutex_lock l(lock_);
  region_visitors_.push_back(visitor);
  if (base_ptr_ != nullptr) {
    visitor(base_ptr_, gpu_memory_size_);
  }
}

bool GPUBFCAllocator::TracksAllocationSizes() { return true; }

size_t GPUBFCAllocator::RequestedSize(void* ptr) {
  mutex_lock l(lock_);
  auto it = ptr_to_chunk_map_.find(ptr);
  CHECK(it != ptr_to_chunk_map_.end())
      << "Asked for requested size of pointer we never allocated: " << ptr;
  GPUBFCAllocator::Chunk* c = it->second;
  return c->requested_size;
}

size_t GPUBFCAllocator::AllocatedSize(void* ptr) {
  mutex_lock l(lock_);
  auto it = ptr_to_chunk_map_.find(ptr);
  CHECK(it != ptr_to_chunk_map_.end())
      << "Asked for allocated size of pointer we never allocated: " << ptr;
  GPUBFCAllocator::Chunk* c = it->second;
  return c->size;
}

int64 GPUBFCAllocator::AllocationId(void* ptr) {
  mutex_lock l(lock_);
  auto it = ptr_to_chunk_map_.find(ptr);
  CHECK(it != ptr_to_chunk_map_.end())
      << "Asked for allocation id of pointer we never allocated: " << ptr;
  GPUBFCAllocator::Chunk* c = it->second;
  return c->allocation_id;
}

void GPUBFCAllocator::DumpMemoryLog(size_t num_bytes) {
  // For each bin: tally up the total number of chunks and bytes.
  // Note that bins hold only free chunks.
  for (auto bit : bins_) {
    Bin* b = bit.second;

    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_requested_bytes_in_use = 0;
    size_t total_requested_bytes_in_bin = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
    for (Chunk* c : b->free_chunks) {
      total_bytes_in_bin += c->size;
      total_requested_bytes_in_bin += c->requested_size;
      ++total_chunks_in_bin;
      if (c->in_use()) {
        total_bytes_in_use += c->size;
        total_requested_bytes_in_use += c->requested_size;
        ++total_chunks_in_use;
      }
    }

    LOG(INFO) << "Bin (" << b->bin_size
              << "): \tTotal Chunks: " << total_chunks_in_bin
              << ", Chunks in use: " << total_chunks_in_use << " "
              << strings::HumanReadableNumBytes(total_bytes_in_bin)
              << " allocated for chunks. "
              << strings::HumanReadableNumBytes(total_requested_bytes_in_bin)
              << " client-requested for chunks. "
              << strings::HumanReadableNumBytes(total_bytes_in_use)
              << " in use in bin. "
              << strings::HumanReadableNumBytes(total_requested_bytes_in_use)
              << " client-requested in use in bin.";
  }

  // Find the bin that we would have liked to allocate in, so we
  // can get some further analysis about fragmentation.
  auto it = bins_.lower_bound(num_bytes);
  if (it != bins_.end()) {
    Bin* b = it->second;

    LOG(INFO) << "Bin for " << strings::HumanReadableNumBytes(num_bytes)
              << " was " << strings::HumanReadableNumBytes(b->bin_size)
              << ", Chunk State: ";

    for (Chunk* c : b->free_chunks) {
      LOG(INFO) << c->DebugString(true);
    }
  }

  // Next show the chunks that are in use, and also summarize their
  // number by size.
  std::map<size_t, int> in_use_by_size;
  for (auto& it : ptr_to_chunk_map_) {
    const Chunk& c = *it.second;
    in_use_by_size[c.size]++;
    LOG(INFO) << "Chunk at " << it.first << " of size " << c.size;
  }

  LOG(INFO) << "     Summary of in-use Chunks by size: ";
  size_t total_bytes = 0;
  for (auto& it : in_use_by_size) {
    LOG(INFO) << it.second << " Chunks of size " << it.first << " totalling "
              << strings::HumanReadableNumBytes(it.first * it.second);
    total_bytes += (it.first * it.second);
  }
  LOG(INFO) << "Sum Total of in-use chunks: "
            << strings::HumanReadableNumBytes(total_bytes);
}
}  // namespace tensorflow
