// Sam Whitlock (sam.whitlock@epfl.ch)

#ifndef TENSORFLOW_KERNELS_READER_ASYNC_BASE_H_
#define TENSORFLOW_KERNELS_READER_ASYNC_BASE_H_

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <utility>
#include "tensorflow/core/framework/reader_interface.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class ReaderAsyncBase : public ReaderInterface
{
public:
  class InputChunk
  {
  public:
    InputChunk(std::shared_ptr<ReadOnlyMemoryRegion> &file_region,
               std::size_t offset, std::size_t length) :
        base_file_(file_region), length_(length)
    {
      data_ = file_region->data + offset;
      if (offset+length >= file_region->length) {
        // TODO print some log warning here!
      }
    }

    void GetChunk(const void** data, std::size_t *length);
    {
      *data = data_;
      *length = length_;
    }

  private:
    const void* data_;
    std::size_t length_;
    std::shared_ptr<ReadOnlyMemoryRegion> base_file_;
  };

  explicit ReaderAsyncBase();

  // Fill the buffer from the next Chunk in the queue
  // by calling GetNextChunk
  // The buffer will be empty before the call is made.
  virtual Status FillBuffer(vector<char> &buffer) = 0;

  virtual Status ChunkWorkItem(const string &filename) = 0;

// TODO need methods for reading a single item

protected:
  Status EnqueueNextChunk(InputChunk &&input_chunk);

  Status GetNextWork(InputChunk *next_chunk);

private:

  // Implementations of ReaderInterface methods.  These ensure thread-safety
  // and call the methods above to do the work.
  void Read(QueueInterface* queue, string* key, string* value,
            OpKernelContext* context) override;
  void ReadBatch(QueueInterface* queue, 
                 std::function<string*(int)> batch_loader, 
                 int batch_size, string* key, OpKernelContext* context,
                 int* produced) override;
  Status Reset() override;
  int64 NumRecordsProduced() override;
  int64 NumWorkUnitsCompleted() override;
  Status SerializeState(string* state) override;
  Status RestoreState(const string& state) override;

  std::unique_ptr<thread::ThreadPool> thread_pool_;
  ObjectPool<std::vector<char>> buffer_pool_;

  // Member variables for mutating the Chunk queue
  std::deque<InputChunk> chunk_queue_;
  mutable std::mutex chunk_queue_mu_;
  mutable std::condition_variable chunk_queue_cv_;
};

} // namespace tensorflow {

#endif TENSORFLOW_KERNELS_READER_ASYNC_BASE_H_
