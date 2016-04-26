
// Stuart Byma
// Mostly copied from reader_interface.h


#ifndef TENSORFLOW_FRAMEWORK_WRITER_INTERFACE_H_
#define TENSORFLOW_FRAMEWORK_WRITER_INTERFACE_H_

#include <memory>
#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class WriterInterface;

// Readers are the mechanism for reading records from files in
// TensorFlow graphs.  Each supported file format has a corresponding
// WriterInterface descendant and a corresponding Op & OpKernel
// (implemented using ReaderOpKernel from reader_op_kernel.h).
//
// To use a Reader, you first encode "work" (some string, typically a
// filename) in the Reader's "work queue".  It then processes the
// "work" (reading records from the file), to produce key/value
// strings.  The methods of this class are called by ReaderFoo ops,
// so see ../ops/io_ops.cc for detailed descriptions.
//
// All descendants of this class must be thread-safe.
class WriterInterface : public ResourceBase {
 public:

  // Using the list of tensors `values`, format data
  // and write to file using associated key.
  virtual void Write(OpInputList* values, string key, 
      OpKernelContext* context) = 0;

  // close any open files,etc. Normally called at end of session 
  virtual void Done(OpKernelContext* context) = 0;
  // Accessors
  virtual int64 NumRecordsProduced() = 0;

  // -- Serialization/Restoration support --
  // Not all readers will support saving and restoring state.
  virtual Status SerializeState(string* state) = 0;
  // Note: Must Reset on error.
  virtual Status RestoreState(const string& state) = 0;

  string DebugString() override { return "a writer"; }

 protected:
  virtual ~WriterInterface() {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_WRITER_INTERFACE_H_
