
// Stuart Byma
// Mostly copied from reader_base.cc

#include "tensorflow/core/kernels/writer_base.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

// WriterBase ------------------------------------------------------

WriterBase::WriterBase(const string& name, const string& work) 
    : name_(name), work_(work) {}

int64 WriterBase::NumRecordsProduced() {
  mutex_lock lock(mu_);
  return num_records_produced_;
}

Status WriterBase::SerializeState(string* state) {
  mutex_lock lock(mu_);
  return SerializeStateLocked(state);
}

Status WriterBase::SerializeStateLocked(string* state) {
  return errors::Unimplemented("Writer SerializeState");
}

Status WriterBase::RestoreState(const string& state) {
  mutex_lock lock(mu_);
  Status status = RestoreStateLocked(state);
  /*if (!status.ok()) {  // should still have reset?
    ResetLocked();
  }*/
  return status;
}

Status WriterBase::RestoreStateLocked(const string& state) {
  return errors::Unimplemented("Writer RestoreState");
}

void WriterBase::Done(OpKernelContext* context) {
  mutex_lock lock(mu_);

  /*if (!work_in_progess()) {
    context->SetStatus(errors::Internal("Tried to call Done",
                  " on Writer with no work in progress"));
    return;
  }*/

  Status status = OnWorkFinishedLocked();
  if (!status.ok()) {
    context->SetStatus(status);
  }

  return;
}

void WriterBase::Write(const string* value,
                      OpKernelContext* context) {
  mutex_lock lock(mu_);

  if (!work_in_progress()) {
    ++work_started_;
    Status status = OnWorkStartedLocked(context);
    if (!status.ok()) {
      context->SetStatus(status);
      --work_started_;
    }
  }

  Status status = WriteLocked(*value);
  
  if (!status.ok()) {
    context->SetStatus(errors::Internal(
                  "WriteLocked failed to write value"));
    return;
  }

  ++num_records_produced_; 
  return;
}

void WriterBase::SaveBaseState(WriterBaseState* state) const {
  state->Clear();
  state->set_work_started(work_started_);
  state->set_work_finished(work_finished_);
  state->set_num_records_produced(num_records_produced_);
  state->set_current_work(work_);
}

Status WriterBase::RestoreBaseState(const WriterBaseState& state) {
  work_started_ = state.work_started();
  work_finished_ = state.work_finished();
  num_records_produced_ = state.num_records_produced();
  work_ = state.current_work();
  if (work_started_ < 0 || work_finished_ < 0 || num_records_produced_ < 0) {
    return errors::InvalidArgument(
        "Unexpected negative value when restoring in ", name(), ": ",
        state.DebugString());
  }
  if (work_started_ > work_finished_) {
    return errors::InvalidArgument(
        "Inconsistent work started vs. finished when restoring in ", name(),
        ": ", state.DebugString());
  }
  return Status::OK();
}

}  // namespace tensorflow
