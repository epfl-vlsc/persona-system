#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/file_system.h"
#include "scope_timer.h"
#include <rados/librados.hpp>

namespace tensorflow {

  using namespace std;

  REGISTER_OP("FileStorage")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("cluster_name: string = ''")
  .Attr("cluster_user: string = ''")
  .Input("queue_handle: Ref(string)")
  .Output("file_handle: string")
  .Output("file_name: string")
  .SetIsStateful()
  .Doc(R"doc(
Fetches files from storage, synchronously reads them, and produces a Tensor<2>
with the container and shared name for the file.

queue_handle: a handle to the filename queue
file_handle: a Tensor(2) of strings to access the file resource in downstream nodes
file_name: a Tensor() of string for the unique key for this file
  )doc");

  class FileStorageOp : public OpKernel {
  public:
    FileStorageOp(OpKernelConstruction* context) : OpKernel(context) {};

    void Compute(OpKernelContext* ctx) override {
      if(!initialized_cluster_) {
        LOG(INFO) << "Initializing rados cluster";

        string cluster_name;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
        string cluster_user;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_user", &cluster_user));
      }

      QueueInterface* queue;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "queue_handle", &queue));

      string filename;
      OP_REQUIRES_OK(ctx, GetNextFilename(queue, &filename, ctx));

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));

      auto creator = [filename, ctx](MemoryMappedFile **mmf) {
        ReadOnlyMemoryRegion *rmr;
        TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile(filename, &rmr));
        shared_ptr<ReadOnlyMemoryRegion> shared_rmr(rmr);
        *mmf = new MemoryMappedFile(shared_rmr);
        return Status::OK();
      };

      MemoryMappedFile *mmf;
      OP_REQUIRES_OK(ctx,
                     cinfo.resource_manager()->LookupOrCreate<MemoryMappedFile>(
                       cinfo.container(), filename, &mmf, creator));

      Tensor *output_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_handle", TensorShape({1, 2}), &output_tensor));

      auto output_matrix = output_tensor->matrix<string>();
      output_matrix(0, 0) = cinfo.container();
      output_matrix(0, 1) = filename;

      Tensor *file_name;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_name", TensorShape({1}), &file_name));

      auto scalar = file_name->vec<string>();
      scalar(0) = filename;
    }

  private:
    bool initialized_cluster_ = false;
    librados::Rados cluster;

    Status GetNextFilename(QueueInterface *queue, string *filename, OpKernelContext *ctx) {
      Notification n;
      queue->TryDequeue(ctx, [ctx, &n, filename](const QueueInterface::Tuple& tuple) {
        if (ctx->status().ok()) {
          if (tuple.size() != 1) {
            ctx->SetStatus(errors::InvalidArgument("Expected single component queue"));
          } else if (tuple[0].dtype() != DT_STRING) {
            ctx->SetStatus(errors::InvalidArgument("Expected queue with single string component"));
          } else if (tuple[0].NumElements() != 1) {
            ctx->SetStatus(errors::InvalidArgument("Expected to dequeue a one-element string tensor"));
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
  };

  REGISTER_KERNEL_BUILDER(Name("FileStorage").Device(DEVICE_CPU), FileStorageOp);

} // namespace tensorflow
