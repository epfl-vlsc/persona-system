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
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/object_pool.h"
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
                 std::size_t offset, std::size_t length);
    InputChunk();

    void GetChunk(const void** data, std::size_t *length);

  private:
    const void* data_;
    std::size_t length_;
    std::shared_ptr<ReadOnlyMemoryRegion> base_file_;
  };

  explicit ReaderAsyncBase(int parallel_fill = 1, int buffer_factor = 1);

  // Fill the buffer from the next Chunk in the queue
  // by calling GetNextChunk
  // The buffer will be empty before the call is made.
  virtual Status FillBuffer(InputChunk *chunk, std::vector<char> &buffer) = 0;

/*
  For the given filename, insert one or more InputChunks into the queue
  with EnqueueNextChunk
 */
  virtual Status ChunkWorkItem(const string &filename) = 0;

  virtual Status ReadLocked(string *key, string *value, bool *produced) = 0;

  virtual Status ReadBatchLocked(Tensor* batch_tensor, string *key, int* num_produced);

  virtual Status ResetLocked();

protected:
  Status EnqueueNextChunk(InputChunk &&input_chunk);

  bool GetCurrentBuffer(const std::vector<char> **buf);

  virtual TensorShape GetRequiredShape() override;
  virtual DataType GetRequiredType() override;

private:

  // Implementations of ReaderInterface methods.  These ensure thread-safety
  // and call the methods above to do the work.
  void Read(QueueInterface* queue, string* key, string* value,
            OpKernelContext* context) override;

  // Read `batch_size` records. `batch_loader` returns a pointer
  // to string for a given batch index. 
  // Set status on *context with OutOfRange if the current work
  // is complete and the queue is done. In this case, a full
  // batch may not be processed, but partial batches are still valid.
  void ReadBatch(QueueInterface* queue,
                         Tensor* batch_tensor, string* key, OpKernelContext* context,
                         int* produced) override;


  Status Reset() override;
  int64 NumRecordsProduced() override;
  int64 NumWorkUnitsCompleted() override;
  Status SerializeState(string* state) override;
  Status RestoreState(const string& state) override;
  Status InitializeOnce(QueueInterface *queue, OpKernelContext *context) override;

  // Some methods for our own use
  Status GetNextLoan();
  Status GetNextInputChunk(InputChunk *next_chunk);

  void BufferFillerThread(OpKernelContext *context);
  void EnqueueThread(QueueInterface *queue, OpKernelContext *context);

  volatile bool run_ = true;
  bool initialized_ = false;
  int num_threads_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  ObjectPool<std::vector<char>> buffer_pool_;
  decltype(buffer_pool_)::ObjectLoan current_loan_;

  // Member variables for mutating the Chunk queue
  std::deque<InputChunk> chunk_queue_;
  // TODO may need to be a std mutex?
  mutable mutex chunk_queue_mu_;
  mutable std::condition_variable chunk_queue_cv_;

  mutable mutex mu_;

  // Accounting stuff for ReaderInterface overrides
  int64 num_records_produced_ = 0;
  int64 work_finished_ = 0;

  // Both of these are shared between the GetNextLoan (main) thread
  // and the chunking thread
  volatile bool chunking_done_ = false;
  volatile uint64_t chunks_produced_ = 0;
  volatile uint64_t chunks_consumed_ = 0;
};

} // namespace tensorflow {

#endif // TENSORFLOW_KERNELS_READER_ASYNC_BASE_H_
