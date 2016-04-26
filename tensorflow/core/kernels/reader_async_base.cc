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
  data_ = file_region->data + offset;
  if (offset+length >= file_region->length) {
    // TODO print some log warning here!
  }
}

ReaderAsyncBase::InputChunk::GetChunk(const void** data, std::size_t *length)
{
  *data = data_;
  *length = length_;
}

ReaderAsyncBase::ReaderAsyncBase(int parallel_fill, int buffer_factor) :
  buffer_pool_(parallel_fill * buffer_factor, []{ return new vector<char>()}),
  num_threads_(parallel_fill+1) // +1 for the background thread that fills the work chunks
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
          chunk_queue_cv_.empty() && run_;
        });
    }

    if (run_) { // need to double check: could have been aborted
      *next_chunk = std::move(chunk_queue_cv_.pop_front());
      return Status::OK();
    }
  }

  return errors::ResourceExhausted("reader_async_base has been shutdown: GetNextInputChunk");
}

// All the overridden functions from ReaderInterface

void ReaderAsyncBase::Read(QueueInterface* queue, string* key, string* value,
          OpKernelContext* context) override
{
  mutex_lock l(mu_);
  Status status;
  if (!run_) {
    context->setStatus(errors::Unavailable("ReaderAsyncBase::Read - run_ is not active when Read is called"));
    return;
  }

  if (current_loan_.get() == nullptr) {
    status = GetNextLoan();
    if (!status.ok()) {
      context->setStatus(status);
      return;
    }
  }

  bool produced = false;
  while (!produced) {
    status = ReadLocked(key, value, &produced);
    if (status.ok()) {
      if (!produced) {
        context->SetStatus(errors::Internal("ReadLocked() returned OK, but not produced"));
        return;
      }
    } else {
      if (errors::IsResourceExhausted(status)) {
        if (produced) {
          context->SetStatus(errors::Internal("ReadLocked() returned error, but produced output"));
          return;
        } else {
          status = GetNextLoan();
          if (status.ok()) {
            continue;
          }
        }
      }

      context->SetStatus(status);
      return;
    }
  }

}

void ReaderAsyncBase::ReadBatch(QueueInterface* queue,
               std::function<string*(int)> batch_loader,
               int batch_size, string* key, OpKernelContext* context,
               int* produced) override
{
  mutex_lock l(mu_);
  Status status;
  if (!run_) {
    context->setStatus(errors::Unavailable("ReaderAsyncBase::Read - run_ is not active when Read is called"));
    return;
  }

  if (current_loan_.get() == nullptr) {
    status = GetNextLoan();
    if (!status.ok()) {
      context->setStatus(status);
      return;
    }
  }

  int num_produced, total_num_produced = 0;
  while (true) {
    num_produced = 0;
    status = ReadBatchLocked(batch_loader, key, batch_size, &num_produced);
    if (status.ok()) {
      break;
    } else {
      if (errors::IsResourceExhausted(status)) {
        total_num_produced += num_produced;
        batch_size -= num_produced;
        status = GetNextLoan();
        if (status.ok()) {
          total_num_produced += num_produced;
          batch_size -= num_produced;
          continue;
        }
      }
      context->SetStatus(status);
      return;
    }
  }
  *produecd = total_num_produced;
}

Status ReaderAsyncBase::GetNextLoan()
{
  if (chunking_done_ && chunks_produced_ == chunks_consumed_) {
    *exhausted = true;
    return errors::ResourceExhausted("GetNextLoan(): No more chunks are available");
  }
  current_loan_.ReleaseEmpty();
  current_loan_ = object_pool_.GetReady();
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
      if (IsResourceExhausted(s)) {
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

    s = FillBuffer(&work, &empty_buf);
    if (!s.is_ok()) {
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

Status ReaderAsyncBase::Reset() override
{
  mutex_lock l(mu_);
  return ResetLocked();
}

Status ReaderAsyncBase::ResetLocked()
{
  work_finished_ = 0;
  num_records_produced_ = 0;
  chunks_produced_ = 0;
  chunks_consumed_ = 0;
  chunking_done_ = false;
  // TODO what to do about run_ and initialized_ ?
  return Status::OK();
}

int64 ReaderAsyncBase::NumRecordsProduced() override
{
  mutex_lock l(mu_);
  return num_records_produced_;
}

int64 ReaderAsyncBase::NumWorkUnitsCompleted() override
{
  mutex_lock l(mu_);
  return work_finished_;
}

// Just.....no. Not right now :)
Status ReaderAsyncBase::SerializeState(string* state) override
{ return errors::Unimplemented("Async Reader SerializeState"); }
Status ReaderAsyncBase::RestoreState(const string& state) override
{ return errors::Unimplemented("Async Reader RestoreState"); }

virtual Status ReadBatchLocked(std::function<string*(int)> batch_loader,
                                int num_requested,
                                int *num_produced) {
  return errors::Unimplemented("Async Reader ReadBatchLocked"); 
}

Status ReaderAsyncBase::Initialize(OpKernelContext *context)
{
  using namespace thread;
  if (!initialized_) {
    if (num_threads_ < 2) {
      return errors::Internal("Threadpool for ReaderAsyncBase must have at least 2 threads to make progress!");
    }

    thread_pool_ = unique_ptr<ThreadPool>(new ThreadPool(context->env(), "reader_async_thread_", num_threads_));
    int buffer_filling_threads = num_threads_-1;
    auto thread_closure = [this, context] { BufferFillerThread(context); };
    for (int i = 0; i < buffer_filling_threads; i++) {
      thread_pool_->Schedule(thread_closure);
    }
    initialized_ = true;
  } else {
    // TODO log a warning here
  }

  return Status::OK();
}

void ReaderAsyncBase::EnqueueThread(QueueInterface *queue, OpKernelContext *context)
{
  core::ScopedUnref unref_queue(queue);
  core::ScopedUnref unref_ctx(context);

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
        ++work_started_; // TODO what to do with this?
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

Status ReaderAsyncBase::InitializeOnce(QueueInterface *queue, OpKernelContext *context) override
{
  queue->Ref();
  context->Ref(); // TODO is this necessary?

  auto enqueue_thread = [this, queue, context] {
    EnqueueThread(queue, context);
  };

  thread_pool_->Schedule(std::move(enqueue_thread));
  return Status::OK();
}

} // namespace tensorflow {
