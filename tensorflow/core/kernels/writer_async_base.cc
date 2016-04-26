
// Stuart Byma
// Mostly copied from reader_base.cc

#include "tensorflow/core/kernels/writer_async_base.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {


// WriterAsyncBase ------------------------------------------------------

WriterAsyncBase::WriterAsyncBase(const string& name, const string& work, 
    int num_buffers, uint64 buffer_size)
    : name_(name), work_(work), buffer_size_(buffer_size) {

  buf_pool_ = new BufferPool(buffer_size, num_buffers);
  finish_ = false;

}

WriterAsyncBase::~WriterAsyncBase() { 
  delete buf_pool_; 
}

int64 WriterAsyncBase::NumRecordsProduced() {
  mutex_lock lock(mu_);
  return num_records_produced_;
}

Status WriterAsyncBase::SerializeState(string* state) {
  mutex_lock lock(mu_);
  return SerializeStateLocked(state);
}

Status WriterAsyncBase::SerializeStateLocked(string* state) {
  return errors::Unimplemented("Writer SerializeState");
}

Status WriterAsyncBase::RestoreState(const string& state) {
  mutex_lock lock(mu_);
  Status status = RestoreStateLocked(state);
  /*if (!status.ok()) {  // should still have reset?
    ResetLocked();
  }*/
  return status;
}

Status WriterAsyncBase::RestoreStateLocked(const string& state) {
  return errors::Unimplemented("Writer RestoreState");
}

void WriterAsyncBase::Done(OpKernelContext* context) {
  mutex_lock lock(mu_);
  LOG(INFO) << name_ << ": Called done, emtpying buffer pool...";

  if (!initialized_) {
      LOG(INFO) << name_ << ": Was not initialized; exiting.";
      return;
  }

  while (!buf_pool_->IsReadyEmpty())
    finish_ = false;

  finish_ = true;

  while (!buf_pool_->IsAvailableEmpty())
    finish_ = true;
  /*if (!work_in_progess()) {
    context->SetStatus(errors::Internal("Tried to call Done",
                  " on Writer with no work in progress"));
    return;
  }*/

  Status status = OnWorkFinishedLocked();
  if (!status.ok()) {
    context->SetStatus(status);
  }
  status = outfile_->Close();
  if (!status.ok()) {
    context->SetStatus(status);
  }

  return;
}

void WriterAsyncBase::Write(OpInputList* values, string key,
                      OpKernelContext* context) {

  {
    // make sure this only happens once
    mutex_lock lock(mu_);

    if (!initialized_) {
      Status status = OnWorkStartedLocked(context, &outfile_);
      initialized_ = true;
      if (!status.ok()) {
        LOG(INFO) << "Error: writerasyncbase onworkstartedlocked failed";
        context->SetStatus(status);
      }
      LOG(INFO) << "initializing async writer threadpool";
      thread_pool_ = unique_ptr<thread::ThreadPool>(new thread::ThreadPool(
            context->env(), "writer_async_thread_", 1 /* num_threads */)); 

      auto writer = [this] () {
        BufferPool::Buffer* ready_buf;
        // while not finished, loop thru buffers and write to file
        while (!finish_) {
          ready_buf = buf_pool_->GetNextReady();
          if (!ready_buf)
            continue;
          else {
            char* buffer = ready_buf->GetBuffer();
            StringPiece to_write(buffer, ready_buf->Used());
            //LOG(INFO) << "Appending data in writer thread: " << to_write;
            Status status = outfile_->Append(to_write);
            if (!status.ok()) {
              LOG(INFO) << "WRITER THREAD GOT IO ERROR! EXITING!!!";
              break;
            }
            // reset the buffer and make available
            ready_buf->Reset();
            buf_pool_->BufferAvailable(ready_buf);
          }
        }
        // write all remaining data to disk
        while ((ready_buf = buf_pool_->GetNextAvailable())) {
          char* buffer = ready_buf->GetBuffer();
          if (ready_buf->Used() == 0)
            continue;
          StringPiece to_write(buffer, ready_buf->Used());
          //LOG(INFO) << "Appending data in writer thread: " << to_write;
          Status status = outfile_->Append(to_write);
          if (!status.ok()) {
            LOG(INFO) << "WRITER THREAD GOT IO ERROR! EXITING!!!";
            break;
          }
        }
        LOG(INFO) << name_ << ": Writer thread is ending...";
      };

      thread_pool_->Schedule(writer);
    }
  } // mutex lock

  // find an available buffer with space
  BufferPool::Buffer* buf;
  while (!(buf = buf_pool_->GetNextAvailable())) {;;}

  uint64 used = 0;
  Status status = WriteUnlocked(values, key, buf->GetCurrentBuffer(), 
      buf->GetCurrentBufferSize(), &used);

  int count = 0;
  while (errors::IsResourceExhausted(status)) {
    
    if (count > buf_pool_->NumBuffers()*2) {
      LOG(INFO) << "failed to write after checking all buffers twice"
        << ", buffer too small." <<
        " You should increase the buffer size.";
      context->SetStatus(status);
      buf_pool_->BufferAvailable(buf);
      return;
    }
    // buffer was too full. Hand off for write and get a fresh one
    /*LOG(INFO) << "Resource was exhausted, only " << buf->GetCurrentBufferSize()
      << " bytes left, moving to new buffer";*/
    buf_pool_->BufferReady(buf);
    while (!(buf = buf_pool_->GetNextAvailable())) {;;}
    /*LOG(INFO) << "calling again with cur buf size = " << 
      buf->GetCurrentBufferSize() << " bytes";*/
    status = WriteUnlocked(values, key, buf->GetCurrentBuffer(), 
        buf->GetCurrentBufferSize(), &used);
    count++;

  }

  if (status.ok() && used == 0) {
    //LOG(INFO) << "result used 0 bytes of buffer!!!";
    // the users writer basically ignored this value
    // forget about it
    buf_pool_->BufferAvailable(buf);
    return;
  }

  if (!status.ok()) {
    buf_pool_->BufferAvailable(buf); // recycle the buffer
    LOG(INFO) << "WriteUnlocked status was not OK!!";
    context->SetStatus(errors::Internal(
                  "WriteUnlocked failed to write value"));
    return;
  }
  
  // update the amount of the buffer used
  /*LOG(INFO) << "Setting used = " << used << " on buffer with "
    << buf->GetCurrentBufferSize() << " bytes remaining";*/
  buf->SetUsed(used);
  buf_pool_->BufferAvailable(buf); // all OK, return buf to pool

  ++num_records_produced_;
  return;
}

void WriterAsyncBase::SaveBaseState(WriterAsyncBaseState* state) const {
  state->Clear();
  state->set_num_records_produced(num_records_produced_);
  state->set_current_work(work_);
}

Status WriterAsyncBase::RestoreBaseState(const WriterAsyncBaseState& state) {
  num_records_produced_ = state.num_records_produced();
  work_ = state.current_work();
  return Status::OK();
}

}  // namespace tensorflow
