#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/file_system.h"
#include <rados/librados.hpp>
#include "shared_mmap_file_resource.h"

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

  class StringReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
  public:
    StringReadOnlyMemoryRegion(std::string str) : str_(str) {}

    const void* data() override { return static_cast<void*>(&str_); }
    uint64 length() override { return str_.size(); }

  private:
    std::string str_;
  }

  class FileStorageOp : public OpKernel {
  public:
    FileStorageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      LOG(INFO) << "Initializing rados cluster";

      string cluster_name;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
      string cluster_user;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_user", &cluster_user));

      int retval;

      retval = cluster.init2(cluster_user.c_str(), cluster_name.c_str(), 0); // TODO flags?
      if(retval < 0) {
        LOG(FATAL) << "Error in cluster.init2: " << retval;
      }

      retval = cluster.connect();
      if(retval < 0) {
        LOG(FATAL) << "Error in cluster.connect: " << retval;
      }

      retval = cluster.ioctx_create(pool_name, io_ctx);
      if(retval < 0) {
        LOG(FATAL) << "Error in cluster.ioctx_create: " << retval;
      }
    }

    void Compute(OpKernelContext* ctx) override {
      QueueInterface* queue;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "queue_handle", &queue));

      string filename;
      OP_REQUIRES_OK(ctx, GetNextFilename(queue, &filename, ctx));

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));

      auto creator = [this, filename](MemoryMappedFile **mmf) {
        librados::bufferlist read_buf;
        librados::AioCompletion *read_completion = librados::Rados::aio_create_completion();
        int retval;

        retval = io_ctx.aio_read(filename, read_completion, &read_buf, read_len, 0);
        if(retval < 0) {
          return errors::Unknown("Error during io_ctx.aio_read: ", to_string(retval));
        }

        // TODO: This is awful; asynchrony would be nice.
        read_completion->wait_for_complete();
        retval = read_completion->get_return_value();
        if(retval < 0) {
          return errors::Unknown("Error during read_completion->get_return_value: ", to_string(retval));
        }

        shared_ptr<ReadOnlyMemoryRegionn> rmr(new StringReadOnlyMemoryRegion(read_buf.to_str()));
        *mmf = new MemoryMappedFile(rmr);
        return Status::OK();
      };

      MemoryMappedFile *file;
      OP_REQUIRES_OK(ctx,
                     cinfo.resource_manager()->LookupOrCreate<MemoryMappedFile>(
                       cinfo.container(), filename, &file, creator));

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
    librados::Rados cluster;
    librados::IoCtx io_ctx;

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
