
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
  LOG(INFO) << "Called done, emtpying buffer pool...";

  while (!buf_pool_->IsReadyEmpty())
    finish_ = false;

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

void WriterAsyncBase::Write(const string* value,
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
            Status status = outfile_->Append(to_write);
            if (!status.ok()) {
              LOG(INFO) << "WRITER THREAD GOT IO ERROR! EXITING!!!";
              break;
            }
            buf_pool_->BufferEmpty(ready_buf);
          }
        }
        LOG(INFO) << "Writer thread is ending...";
      };

      thread_pool_->Schedule(writer);
    }
  } // mutex lock

  // find a free buffer or block
  BufferPool::Buffer* buf;
  while (!(buf = buf_pool_->GetNextEmpty())) {;;}

  uint64 used = 0;
  Status status = WriteUnlocked(*value, buf->GetBuffer(), 
      buf->GetBufferSize(), &used);
 
  if (used == 0) {
    LOG(INFO) << "result used 0 bytes of buffer!!!";
  }

  if (!status.ok()) {
    buf_pool_->BufferEmpty(buf); // recycle the buffer
    LOG(INFO) << "WriteUnlocked status was not OK!!";
    context->SetStatus(errors::Internal(
                  "WriteUnlocked failed to write value"));
    return;
  }

  buf->SetUsed(used);
  buf_pool_->BufferReady(buf); // all OK, set buf to ready

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
