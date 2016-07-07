#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "data.h"
#include "tensorflow/core/user_ops/dense-format/buffer.h"
#include "libs3.h"
#include "stdlib.h"
#include <iostream>
#include "shared_mmap_file_resource.h"

namespace tensorflow {

  using namespace std;

  REGISTER_OP("S3Reader")
  .Attr("access_key: string")
  .Attr("secret_key: string")
  .Attr("host: string")
  .Attr("bucket: string")
  .Input("queue_handle: Ref(string)")
  .Input("pool_handle: Ref(string)")
  .Output("file_handle: string")
  .Output("file_name: string")
  .SetIsStateful()
  .Doc(R"doc(
Obtains file names from a queue, fetches those files from storage using S3, and writes
them to a buffer from a pool of buffers.

queue_handle: a handle to the filename queue
pool_handle: a handle to the buffer pool
file_handle: a Tensor(2) of strings to access the file resource in downstream nodes
file_name: a Tensor() of string for the unique key for this file
  )doc");

  class S3ReaderOp : public OpKernel {
  public:
    S3ReaderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      LOG(INFO) << "Initializing LibS3 connection";

      // Receive access_key, secret_key, host, bucket via attributes
      OP_REQUIRES_OK(ctx, ctx->GetAttr("access_key", &access_key));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("secret_key", &secret_key));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("host", &host));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("bucket", &bucket));

      // Initialize bucket context and handlers
      bucketContext.hostName = host.c_str();
      bucketContext.bucketName = bucket.c_str();
      bucketContext.protocol = S3ProtocolHTTP;
      bucketContext.uriStyle = S3UriStylePath;
      bucketContext.accessKeyId = access_key.c_str();
      bucketContext.secretAccessKey = secret_key.c_str();

      responseHandler.propertiesCallback = &S3ReaderOp::responsePropertiesCallback;
      responseHandler.completeCallback = &S3ReaderOp::responseCompleteCallback;

      getObjectHandler.responseHandler = S3ReaderOp::responseHandler;
      getObjectHandler.getObjectDataCallback = &S3ReaderOp::getObjectDataCallback;

      // Open connection
      S3_initialize("s3", S3_INIT_ALL, host.c_str());
    }

    void Compute(OpKernelContext* ctx) override {
      // Get files to download from the queue
      QueueInterface* queue;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "queue_handle", &queue));
      core::ScopedUnref unref_me(queue);

      OP_REQUIRES_OK(ctx, GetNextFilename(queue, &file_key, ctx));

      // Get the buffer to write the file into from the buffer pool
      ReferencePool<Buffer> *ref_pool;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "pool_handle", &ref_pool));
      core::ScopedUnref unref_pool(ref_pool);

      ResourceContainer<Buffer> *rec_buffer;
      OP_REQUIRES_OK(ctx, ref_pool->GetResource(&rec_buffer));
      rec_buffer->get()->reset();

      S3_get_object(&bucketContext, file_key.c_str(), NULL, 0, 0, NULL, &getObjectHandler, rec_buffer);

      // Output tensors
      Tensor *output_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_handle", TensorShape({1, 2}), &output_tensor));
      auto output_matrix = output_tensor->matrix<string>();
      output_matrix(0, 0) = rec_buffer->container();
      output_matrix(0, 1) = rec_buffer->name();

      Tensor *file_name;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_name", TensorShape({1}), &file_name));
      auto scalar = file_name->vec<string>();
      scalar(0) = file_key;
    }

  private:
    string access_key;
    string secret_key;
    string host;
    string bucket;
    string file_key;

    S3BucketContext bucketContext;
    S3ResponseHandler responseHandler;
    S3GetObjectHandler getObjectHandler;

    static S3Status responsePropertiesCallback(
                    const S3ResponseProperties *properties,
                    void *callbackData)
    {
      return S3StatusOK;
    }

    static void responseCompleteCallback(
                    S3Status status,
                    const S3ErrorDetails *error,
                    void *callbackData)
    {
      return;
    }

    static S3Status getObjectDataCallback(int bufferSize, const char *buffer, void *callbackData)
    {
      auto buf = (ResourceContainer<Buffer> *) callbackData;
      buf->get()->AppendBuffer(buffer, bufferSize);
      return S3StatusOK;
    }

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

  REGISTER_KERNEL_BUILDER(Name("S3Reader").Device(DEVICE_CPU), S3ReaderOp);

} // namespace tensorflow