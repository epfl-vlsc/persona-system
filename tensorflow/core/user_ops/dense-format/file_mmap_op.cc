#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"

namespace tensorflow {

  using namespace std;

  REGISTER_OP("FileMMap")
  .Input("queue_handle: Ref(string)")
  .Output("file_handle: string") // or is the output string?
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .SetIsStateful()
  .Doc(R"doc(
  )doc");

  class FileMMapOp : public OpKernel {
  public:
    FileMMapOp(OpKernelConstruction* context) : OpKernel(context) {};

    Status GetNextFilename(QueueInterface *queue, string *filename, OpKernelContext *ctx) {
      Notification n;
      queue->TryDequeue(
                        ctx, [this, ctx, &n, filename](const QueueInterface::Tuple& tuple) {
                          if (ctx->status().ok()) {
                            if (tuple.size() != 1) {
                              ctx->SetStatus(
                                                 errors::InvalidArgument("Expected single component queue"));
                            } else if (tuple[0].dtype() != DT_STRING) {
                              ctx->SetStatus(errors::InvalidArgument(
                                                                         "Expected queue with single string component"));
                            } else if (tuple[0].NumElements() != 1) {
                              ctx->SetStatus(errors::InvalidArgument(
                                                                         "Expected to dequeue a one-element string tensor"));
                            } else {
                              *filename = tuple[0].flat<string>()(0);
                            }
                          }
                          n.Notify();
                        });
      n.WaitForNotification();
      if (!ctx->status().ok()) {
        return ctx->status();
      }
      return Status::OK();
    }

    void Compute(OpKernelContext* ctx) override {
      QueueInterface* queue;
      OP_REQUIRES_OK(ctx,
                     GetResourceFromContext(ctx, "queue_handle", &queue));
      core::ScopedUnref unref_me(queue);

      // 1. get a filename
      string filename;
      OP_REQUIRES_OK(ctx, GetNextFilename(queue, &filename, ctx));

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));

      auto creator = [this, filename, ctx](MemoryMappedFile **mmf) {
        ReadOnlyMemoryRegion *rmr;
        TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile(filename, &rmr));
        shared_ptr<ReadOnlyMemoryRegion> shared_rmr(rmr);
        volatile char c = 0;
        PrimeRegion(static_cast<const char*>(rmr->data()), rmr->length(), &c);
        hack += c;
        *mmf = new MemoryMappedFile(shared_rmr);
        return Status::OK();
      };

      MemoryMappedFile *mmf;
      OP_REQUIRES_OK(ctx,
                     cinfo.resource_manager()->LookupOrCreate<MemoryMappedFile>(
                                                                                 cinfo.container(),
                                                                                 filename,
                                                                                 &mmf,
                                                                                 creator
                                                                                 ));

      Tensor *output_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({2}), &output_tensor));
      MappedFileRef container_ref(output_tensor);
      container_ref.SetName(filename);
      container_ref.SetContainer(cinfo.container());
    }

    void PrimeRegion(const char *data, const size_t length, volatile char* c, const size_t step_size=4096) {
      for (size_t i = 0; i < length; i+=step_size) {
        *c += data[i];
      }
    }
  private:
    volatile char hack = 0;
  };

  REGISTER_KERNEL_BUILDER(Name("FileMMap").Device(DEVICE_CPU), FileMMapOp);

} // namespace tensorflow {
