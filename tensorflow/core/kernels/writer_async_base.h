
// Stuart Byma
// Mostly copied from reader_base.h

#ifndef TENSORFLOW_KERNELS_WRITER_ASYNC_BASE_H_
#define TENSORFLOW_KERNELS_WRITER_ASYNC_BASE_H_

#include <memory>
#include <string>
#include <vector>
#include <queue>
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/writer_interface.h"
#include "tensorflow/core/kernels/writer_base.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

using namespace std;

// a small class to manage sharing a pool of buffers
// between different threads. available_ buffers have 
// space to be written into, while ready_ buffers 
// are full to be consumed. 
class BufferPool {
  public:
    class Buffer {
      // a mutable buffer wrapper
      public:
        Buffer(uint64 buffer_size) : buffer_size_(buffer_size) {
          buf_ = new char[buffer_size];
          cur_buf_ = buf_;
          used_ = 0;  // this is an important line of code
        }

        ~Buffer() {
          delete [] buf_;
        }

        char* GetBuffer() { return buf_; }
        char* GetCurrentBuffer() { return cur_buf_; }
        const uint64 GetBufferSize() { return buffer_size_; }
        const uint64 GetCurrentBufferSize() { return buffer_size_ - used_; }

        uint64 Used() { return used_; }
        void SetUsed(uint64 used) { 
          used_ += used; 
          cur_buf_ += used;
          if (cur_buf_ - buf_ > buffer_size_)
            LOG(INFO) << "BUFFER WAS EXCEEDED WTFFF!!";
        }
        void Reset() {
          used_ = 0;
          cur_buf_ = buf_;
        }

      private:
        char* buf_ = nullptr;
        char* cur_buf_ = nullptr;
        const uint64 buffer_size_;
        uint64 used_; // amount of current buffer used
    };

    BufferPool(uint64 buffer_size, int num_buffers) :
      buffer_size_(buffer_size), num_buffers_(num_buffers) {
      // init buffers
      buffers_.reserve(num_buffers);
      for (int i = 0; i < num_buffers; i++) {
        Buffer* buf = new Buffer(buffer_size);
        buffers_.push_back(buf);
        available_.push(buf);
      }
    }

    ~BufferPool() {
      // destroy buffers
      for (auto buf : buffers_) {
        delete buf;
      }
    }

    Buffer* GetNextReady() {
      mutex_lock lock(ready_mu_);
      if (ready_.empty())
        return nullptr;
      else {
        Buffer* ret = ready_.front();
        ready_.pop();
        return ret;
      }
    }
    
    Buffer* GetNextAvailable() {
      mutex_lock lock(mu_);
      if (available_.empty())
        return nullptr;
      else {
        Buffer* ret = available_.front();
        available_.pop();
        return ret;
      }
    }

    void BufferReady(Buffer* buf) {
      mutex_lock lock(ready_mu_);
      if (IsMemberBuffer(buf)) {
        ready_.push(buf);
      } else {
        LOG(INFO) << "Bufferpool Error: tried to BufferReady a"
          << "a non member buffer";
      }
    }

    void BufferAvailable(Buffer* buf) {
      mutex_lock lock(mu_);
      if (IsMemberBuffer(buf)) {
        available_.push(buf);
      } else {
        LOG(INFO) << "Bufferpool Error: tried to BufferAvailable a"
          << "a non member buffer";
      }
    }

    bool IsReadyEmpty() {
      mutex_lock lock(ready_mu_);
      return ready_.empty();
    }
    
    bool IsAvailableEmpty() {
      mutex_lock lock(mu_);
      return available_.empty();
    }
    
    int NumBuffers() { return num_buffers_; }

  private:
    bool IsMemberBuffer(Buffer* buf) {
      if (std::find(buffers_.begin(), buffers_.end(), buf) != buffers_.end())
        return true;
      else
        return false;
    }
    mutable mutex mu_;
    mutable mutex ready_mu_;
    const uint64 buffer_size_;
    const int num_buffers_;
    vector<Buffer*> buffers_;
    queue<Buffer*> ready_;
    queue<Buffer*> available_;
};

// Default implementation of WriterInterface.
class WriterAsyncBase : public WriterInterface {
 public:
  // name: For use in error messages, should mention both the name of
  // the op and the node.
  explicit WriterAsyncBase(const string& name, const string& work,
      int num_buffers, uint64 buffer_size);

  ~WriterAsyncBase();
  // Note that methods with names ending in "Locked" are called while
  // the WriterBase's mutex is held

  // Implement this function in descendants -----------------------------------

  // Format and write the value to a buffer, which is in turn written
  // asynchronously to the file. Must set amount of buffer used
  // in bytes.
  // Async does not guarantee ordering of writes
  // Usage:
  //    Should return a valid status. OK on successful write to file.
  virtual Status WriteUnlocked(const string& value, char* buffer, uint64 buffer_size, uint64* used) = 0;

  // Called when work starts / finishes.
  // Should set and initialize `file`
  virtual Status OnWorkStartedLocked(OpKernelContext* context, WritableFile** file) = 0; 
  virtual Status OnWorkFinishedLocked()  = 0;

  // do we still need to reset writer kernels?
  //virtual Status ResetLocked()  = 0;

  // Default implementation generates an Unimplemented error.
  // See the protected helper methods below.
  virtual Status SerializeStateLocked(string* state);
  virtual Status RestoreStateLocked(const string& state);

  // Accessors ----------------------------------------------------------------

  // Returns the name of the current work item (valid if
  // work_in_progress() returns true).  May change between calls to
  // ReadLocked().
  const string& current_work() const { return work_; }

  // What was passed to the constructor.
  const string& name() const { return name_; }

  // return the max write buffer size
  const uint64 buffer_size() { return buffer_size_; }

 protected:
  // For descendants wishing to implement serialize & restore state.

  // Writes WriterBase state to *state.
  void SaveBaseState(WriterAsyncBaseState* state) const;

  // Restores WriterBase state from state. Assumes state was filled
  // using SaveBaseState() above.
  Status RestoreBaseState(const WriterAsyncBaseState& state);

 private:
  // Implementations of WriterInterface methods.  These ensure thread-safety
  // and call the methods above to do the work.
  void Done(OpKernelContext* context) override;
  void Write(const string* value,
            OpKernelContext* context) override;
  int64 NumRecordsProduced() override;
  Status SerializeState(string* state) override;
  Status RestoreState(const string& state) override;

  // mutex, some calls need to be serialized
  mutable mutex mu_;
  const string name_;
  WritableFile* outfile_;
  bool initialized_ = false;
  int64 num_records_produced_ = 0;
  string work_;
  uint64 buffer_size_;
  BufferPool* buf_pool_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  bool finish_ = false;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_READER_BASE_H_
