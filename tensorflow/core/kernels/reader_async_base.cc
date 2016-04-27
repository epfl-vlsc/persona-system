// Sam Whitlock (sam.whitlock@epfl.ch)

#include <iostream>

#include "reader_async_base.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

ReaderAsyncBase::InputChunk::InputChunk(std::shared_ptr<ReadOnlyMemoryRegion> &file_region,
                                        std::size_t offset, std::size_t length) :
  base_file_(file_region), length_(length)
{
  data_ = file_region->data() + offset;
  if (offset+length >= file_region->length()) {
    LOG(ERROR) << "desired input chunk is larger than memory region!\nRequested end: " <<
      offset+length << ", actual end: " << file_region->length();
  }
}

ReaderAsyncBase::InputChunk::InputChunk() : data_(nullptr), length_(0), base_file_(nullptr) {}

void ReaderAsyncBase::InputChunk::GetChunk(const void** data, std::size_t *length) const
{
  *data = data_;
  *length = length_;
}

void ReaderAsyncBase::InputChunk::SetFileName(std::string &s) {
  filename = s;
}

const std::string& ReaderAsyncBase::InputChunk::GetFileName() const {
  return filename;
}

ReaderAsyncBase::ReaderAsyncBase(int parallel_fill, int buffer_factor) :
  buffer_pool_(parallel_fill * buffer_factor, []{ return new std::vector<char>(); }),
  num_threads_(parallel_fill+1), // +1 for the background thread that fills the work chunks
  current_loan_(&buffer_pool_)
{
  if (parallel_fill < 1) {
    LOG(ERROR) << "Attempting to use inadequate parallel fill: " << parallel_fill << std::endl;
  }
  if (buffer_factor < 1) {
    LOG(ERROR) << "Attempting to use inadequate buffer factor: " << buffer_factor << std::endl;
  }
}

Status ReaderAsyncBase::EnqueueNextChunk(InputChunk &&input_chunk)
{
  if (run_) {
    mutex_lock cql(chunk_queue_mu_);
    chunks_produced_++;
    chunk_queue_.push_back(input_chunk);
    chunk_queue_cv_.notify_one();
    return Status::OK();
  }

  return errors::Internal("reader_async_base has been shutdown: EnqueueNextChunk");
}

Status ReaderAsyncBase::GetNextInputChunk(InputChunk *next_chunk)
{
  if (run_) {
    mutex_lock cql(chunk_queue_mu_);
    if (chunk_queue_.empty()) {
      chunk_queue_cv_.wait(cql, [this]() {
          return chunk_queue_.empty() && run_;
        });
    }

    if (run_) { // need to double check: could have been aborted
      *next_chunk = chunk_queue_.front();
      chunk_queue_.pop_front();
      return Status::OK();
    } else {
      return errors::Cancelled("Can't get next input chunk. ReaderAsyncBase is not running");
    }
  }

  return errors::ResourceExhausted("reader_async_base has been shutdown: GetNextInputChunk");
}

Status ReaderAsyncBase::ReadLocked(string *key, string *value, bool *produced, bool *done_with_buffer) {
  return errors::Internal("Not implemented!!");
}

Status ReaderAsyncBase::ReadBatchLocked(Tensor* batch_tensor, string *key, int* num_produced, bool *done_with_buffer) {
  return errors::Internal("Not implemented!!");
}

// All the overridden functions from ReaderInterface

void ReaderAsyncBase::Read(QueueInterface* queue, string* key, string* value,
          OpKernelContext* context)
{
  mutex_lock l(mu_);
  Status status;
  if (!run_) {
    context->SetStatus(errors::Unavailable("ReaderAsyncBase::Read - run_ is not active when Read is called"));
    return;
  }

  if (!read_in_progress_) {
    status = GetNextLoan();
    if (!status.ok()) {
      context->SetStatus(status);
      return;
    }
  }

  bool produced = false;
  bool done_with_buffer;
  do {
    status = ReadLocked(key, value, &produced, &done_with_buffer);
    if (status.ok()) {
      if (!produced) {
        status = errors::Internal("ReadLocked() returned OK, but not produced");
      }
    } else if (errors::IsResourceExhausted(status)) {
      if (produced) {
        status = errors::Internal("ReadLocked() returned error, but produced output");
      } else {
        status = GetNextLoan();
      }
    }
  } while (!produced && status.ok());

  if (status.ok()) {
    num_records_produced_++;
    read_in_progress_ = !done_with_buffer;
  } else {
    context->SetStatus(status);
  }
}

TensorShape ReaderAsyncBase::GetRequiredShape() {
  return TensorShape({0});
}

DataType ReaderAsyncBase::GetRequiredType() {
  return DT_STRING;
}

void ReaderAsyncBase::ReadBatch(QueueInterface* queue,
                Tensor* batch_tensor, string* key, OpKernelContext* context,
                int* produced)
{
  mutex_lock l(mu_);
  Status status;
  if (!run_) {
    context->SetStatus(errors::Unavailable("ReaderAsyncBase::Read - run_ is not active when Read is called"));
    return;
  }

  if (!read_in_progress_) {
    status = GetNextLoan();
    if (!status.ok()) {
      context->SetStatus(status);
      return;
    }
  }

  int num_produced;
  bool done_with_buffer;
  do {
    num_produced = 0;
    status = ReadBatchLocked(batch_tensor, key, &num_produced, &done_with_buffer);

    if (num_produced < 0) {
      status = errors::Internal("ReaderAsyncBase::ReadBatch: produced a negative number of reads: ", num_produced);
    } else if (status.ok()) {
      if (num_produced > 0) {
        if (num_produced > batch_tensor->NumElements()) {
          status = errors::Internal("ReaderAsyncBase::ReadBatch: Status is ok, but produced more elements than possible! (produce: ",
                                    num_produced, ", max: ", batch_tensor->NumElements(), ")");
        } else if (!done_with_buffer) {
          read_in_progress_ = !done_with_buffer;
        }
      } else {
        status = errors::Internal("ReaderAsyncBase::ReadBatch: Status is ok, but didn't produce anything");
      }
    } else if (errors::IsResourceExhausted(status)) {
      if (num_produced > 0) {
        status = errors::Internal("ReaderAsyncBase::ReadBatch: resource exhausted, but produced ",
                                  num_produced, " elements!");
      } else {
        status = GetNextLoan();
      }
    }
  } while (status.ok() && num_produced <= 0);

  if (status.ok()) {
    *produced = num_produced;
    num_records_produced_ += num_produced;
  } else {
    context->SetStatus(status);
  }
}

Status ReaderAsyncBase::GetNextLoan()
{
  {
    mutex_lock l(chunk_queue_mu_);
    if (chunking_done_ && chunks_produced_ == chunks_consumed_) {
      return errors::ResourceExhausted("GetNextLoan(): No more chunks are available");
    }
  }
  current_loan_.ReleaseEmpty();
  current_loan_ = buffer_pool_.GetReady();
  if (current_loan_.get() == nullptr) {
    return errors::Internal("GetNextLoan(): unable to get valid ready object from pool");
  }

  return Status::OK();
}

void ReaderAsyncBase::BufferFillerThread(OpKernelContext *context)
{
  InputChunk work;
  Status s;
  while (run_)
  {
    s = GetNextInputChunk(&work);
    if (!s.ok()) {
      if (errors::IsResourceExhausted(s)) {
        chunking_done_ = true;
        break; // we're done
      } else {
        run_ = false;
        LOG(ERROR) << "BufferFillerThread received a non-exhausted error";
        context->SetStatus(s);
        continue;
      }
    }
    auto empty_buf = buffer_pool_.GetEmpty();
    if (empty_buf.get() == nullptr) {
      LOG(ERROR) << "BufferFillerThread couldn't get an empty buffer despite blocking!";
      context->SetStatus(errors::Aborted("buffer_pool_ in BufferFillerThread not large enough"));
      run_ = false;
      continue;
    }

    s = FillBuffer(&work, *empty_buf);
    if (!s.ok()) {
      empty_buf.ReleaseEmpty();
      LOG(ERROR) << "FillBuffer failed for for BufferFillerThread";
      context->SetStatus(s);
      run_ = false;
      continue;
    } else {
      empty_buf.ReleaseReady();
      chunks_consumed_++;
    }
  }
  LOG(INFO) << "BufferFillerThread exiting";
}

bool ReaderAsyncBase::GetCurrentBuffer(const std::vector<char> **buf)
{
  if (current_loan_.get() == nullptr) {
    return false;
  } else {
    *buf = current_loan_.get();
    return true;
  }
}

Status ReaderAsyncBase::Reset()
{
  mutex_lock l(mu_);
  return ResetLocked();
}

Status ReaderAsyncBase::ResetLocked()
{
  mutex_lock l(mu_), cql(chunk_queue_mu_);
  work_finished_ = 0;
  num_records_produced_ = 0;
  chunks_produced_ = 0;
  chunks_consumed_ = 0;
  chunking_done_ = false;
  if (!chunk_queue_.empty()) {
    LOG(WARNING) << "Calling ResetLocked() on AsyncReaderBase with " << chunk_queue_.size() << " elements still in the queue!";
    chunk_queue_.clear();
  }
  chunk_queue_cv_.notify_all();

  buffer_pool_.clear();
  current_loan_ = decltype(buffer_pool_)::ObjectLoan(&buffer_pool_);

  return Status::OK();
}

int64 ReaderAsyncBase::NumRecordsProduced()
{
  mutex_lock l(mu_);
  return num_records_produced_;
}

int64 ReaderAsyncBase::NumWorkUnitsCompleted()
{
  mutex_lock l(mu_);
  return work_finished_;
}

// Just.....no. Not right now :)
Status ReaderAsyncBase::SerializeState(string* state)
{ return errors::Unimplemented("Async Reader SerializeState"); }
Status ReaderAsyncBase::RestoreState(const string& state)
{ return errors::Unimplemented("Async Reader RestoreState"); }

void ReaderAsyncBase::EnqueueThread(QueueInterface *queue, OpKernelContext *context)
{
  core::ScopedUnref unref_queue(queue);
  //core::ScopedUnref unref_ctx(context);

  Notification n;
  auto callback = [this, context, &n](const QueueInterface::Tuple& tuple) {
    if (context->status().ok()) {
      if (tuple.size() != 1) {
        context->SetStatus(
                            errors::InvalidArgument("Expected single component queue"));
      } else if (tuple[0].dtype() != DT_STRING) {
        context->SetStatus(errors::InvalidArgument(
                                                    "Expected queue with single string component"));
      } else if (tuple[0].NumElements() != 1) {
        context->SetStatus(errors::InvalidArgument(
                                                    "Expected to dequeue a one-element string tensor"));
      } else {
        const string &work = tuple[0].flat<string>()(0);
        Status chunk_status = ChunkWorkItem(work);
        if (!chunk_status.ok()) {
          context->SetStatus(chunk_status);
        }
      }
    }
    n.Notify();
  };

  while (run_ && context->status().ok()) {
    queue->TryDequeue(context, callback);
    n.WaitForNotification();
  }

  LOG(INFO) << "Async reader: Enqueuing thread exiting" << std::endl;
  if (!run_)
    LOG(INFO) << "run is false" << std::endl;
  if (context->status().ok())
    LOG(INFO) << "context status is not okay" << std::endl;
}

Status ReaderAsyncBase::InitializeOnce(QueueInterface *queue, OpKernelContext *context)
{
  using thread::ThreadPool;
  if (num_threads_ < 2) {
    return errors::Internal("Threadpool for ReaderAsyncBase must have at least 2 threads to make progress!");
  }

  thread_pool_ = std::unique_ptr<ThreadPool>(new ThreadPool(context->env(), "reader_async_thread_", num_threads_));
  int buffer_filling_threads = num_threads_-1;
  auto thread_closure = [this, context] { BufferFillerThread(context); };
  for (int i = 0; i < buffer_filling_threads; i++) {
    thread_pool_->Schedule(thread_closure);
  }

  queue->Ref();
  auto enqueue_thread = [this, queue, context] {
    EnqueueThread(queue, context);
  };
  thread_pool_->Schedule(std::move(enqueue_thread));

  return Status::OK();
}

} // namespace tensorflow {
