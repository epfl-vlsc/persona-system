
// Stuart Byma
// Mostly copied from reader_base.h

#ifndef TENSORFLOW_KERNELS_Writer_BASE_H_
#define TENSORFLOW_KERNELS_WRITER_BASE_H_

#include <memory>
#include <string>
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/writer_interface.h"
#include "tensorflow/core/kernels/writer_base.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

// Default implementation of WriterInterface.
class WriterBase : public WriterInterface {
 public:
  // name: For use in error messages, should mention both the name of
  // the op and the node.
  explicit WriterBase(const string& name, const string& work);

  // Note that methods with names ending in "Locked" are called while
  // the WriterBase's mutex is held

  // Implement this function in descendants -----------------------------------

  // Write the value to file.
  // This is called "Locked" since it is executed under a mutex
  // that serializes all Writer calls.
  // Usage:
  //    Should return a valid status. OK on successful write to file.
  virtual Status WriteLocked(OpInputList* values, string& key) = 0;

  // Called when work starts / finishes.
  virtual Status OnWorkStartedLocked(OpKernelContext* context) = 0; 
  virtual Status OnWorkFinishedLocked()  = 0;

  // do we still need to reset writer kernels?
  //virtual Status ResetLocked()  = 0;

  // Default implementation generates an Unimplemented error.
  // See the protected helper methods below.
  virtual Status SerializeStateLocked(string* state);
  virtual Status RestoreStateLocked(const string& state);

  // Accessors ----------------------------------------------------------------

  // Always true during a call to ReadLocked().
  bool work_in_progress() const { return work_finished_ < work_started_; }

  // Returns the name of the current work item (valid if
  // work_in_progress() returns true).  May change between calls to
  // ReadLocked().
  const string& current_work() const { return work_; }

  // What was passed to the constructor.
  const string& name() const { return name_; }

 protected:
  // For descendants wishing to implement serialize & restore state.

  // Writes WriterBase state to *state.
  void SaveBaseState(WriterBaseState* state) const;

  // Restores WriterBase state from state. Assumes state was filled
  // using SaveBaseState() above.
  Status RestoreBaseState(const WriterBaseState& state);

 private:
  // Implementations of WriterInterface methods.  These ensure thread-safety
  // and call the methods above to do the work.
  void Done(OpKernelContext* context) override;
  void Write(OpInputList* value, string key,
            OpKernelContext* context) override;
  int64 NumRecordsProduced() override;
  Status SerializeState(string* state) override;
  Status RestoreState(const string& state) override;

  mutable mutex mu_;
  const string name_;
  int64 work_started_ = 0;
  int64 work_finished_ = 0;
  int64 num_records_produced_ = 0;
  string work_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_READER_BASE_H_
