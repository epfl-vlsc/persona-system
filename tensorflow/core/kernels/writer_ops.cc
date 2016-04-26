
// Stuart Byma
// Mostly copied from reader_ops.cc


// See docs in ../ops/io_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/writer_interface.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

class WriterVerbSyncOpKernel : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    WriterInterface* writer;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "writer_handle", &writer));
    ComputeWithWriter(context, writer);
    writer->Unref();
  }

 protected:
  virtual void ComputeWithWriter(OpKernelContext* context,
                                 WriterInterface* writer) = 0;
};

class WriterVerbAsyncOpKernel : public AsyncOpKernel {
 public:
  using AsyncOpKernel::AsyncOpKernel;

  explicit WriterVerbAsyncOpKernel(OpKernelConstruction* context)
      : AsyncOpKernel(context),
        thread_pool_(new thread::ThreadPool(
            context->env(), strings::StrCat("writer_thread_",
                                            SanitizeThreadSuffix(def().name())),
            1 /* num_threads */)) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    WriterInterface* writer;
    OP_REQUIRES_OK(context,
                   GetResourceFromContext(context, "writer_handle", &writer));
    thread_pool_->Schedule([this, context, writer, done]() {
      ComputeWithWriter(context, writer);
      writer->Unref();
      done();
    });
  }

 protected:
  virtual void ComputeWithWriter(OpKernelContext* context,
                                 WriterInterface* writer) = 0;

 private:
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class WriterWriteOp : public WriterVerbAsyncOpKernel {
 public:
  using WriterVerbAsyncOpKernel::WriterVerbAsyncOpKernel;

  void ComputeWithWriter(OpKernelContext* context,
                         WriterInterface* writer) override {

    OpInputList values;
    OP_REQUIRES_OK(context, context->input_list("values", &values));
    const Tensor* key;
    OP_REQUIRES_OK(context, context->input("key", &key));
    auto key_scalar = key->scalar<string>();

    writer->Write(&values, key_scalar(), context);
  }
};

REGISTER_KERNEL_BUILDER(Name("WriterWrite").Device(DEVICE_CPU), WriterWriteOp);

class WriterNumRecordsProducedOp : public WriterVerbSyncOpKernel {
 public:
  using WriterVerbSyncOpKernel::WriterVerbSyncOpKernel;

  void ComputeWithWriter(OpKernelContext* context,
                         WriterInterface* writer) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("records_produced",
                                                     TensorShape({}), &output));
    output->scalar<int64>()() = writer->NumRecordsProduced();
  }
};

REGISTER_KERNEL_BUILDER(Name("WriterNumRecordsProduced").Device(DEVICE_CPU),
                        WriterNumRecordsProducedOp);

class WriterDoneOp : public WriterVerbSyncOpKernel {
 public:
  using WriterVerbSyncOpKernel::WriterVerbSyncOpKernel;

  void ComputeWithWriter(OpKernelContext* context,
                         WriterInterface* writer) override {
    writer->Done(context);
  }
};

REGISTER_KERNEL_BUILDER(Name("WriterDone").Device(DEVICE_CPU),
                        WriterDoneOp);

class WriterSerializeStateOp : public WriterVerbSyncOpKernel {
 public:
  using WriterVerbSyncOpKernel::WriterVerbSyncOpKernel;

  void ComputeWithWriter(OpKernelContext* context,
                         WriterInterface* writer) override {
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("state", TensorShape({}), &output));
    OP_REQUIRES_OK(context,
                   writer->SerializeState(&output->scalar<string>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("WriterSerializeState").Device(DEVICE_CPU),
                        WriterSerializeStateOp);

class WriterRestoreStateOp : public WriterVerbSyncOpKernel {
 public:
  using WriterVerbSyncOpKernel::WriterVerbSyncOpKernel;

  void ComputeWithWriter(OpKernelContext* context,
                         WriterInterface* writer) override {
    const Tensor* tensor;
    OP_REQUIRES_OK(context, context->input("state", &tensor));
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(tensor->shape()),
        errors::InvalidArgument("Writer state must be scalar, but had shape: ",
                                tensor->shape().DebugString()));
    OP_REQUIRES_OK(context, writer->RestoreState(tensor->scalar<string>()()));
  }
};

REGISTER_KERNEL_BUILDER(Name("WriterRestoreState").Device(DEVICE_CPU),
                        WriterRestoreStateOp);


}  // namespace tensorflow
