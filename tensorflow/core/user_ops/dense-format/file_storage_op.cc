#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/file_system.h"
#include "libs3.h"
#include "stdlib.h"
#include <iostream>
#include <fstream>
#include "shared_mmap_file_resource.h"

namespace tensorflow {

  using namespace std;

  REGISTER_OP("FileStorage")
  .Attr("access_key: string")
  .Attr("secret_key: string")
  .Attr("host: string")
  .Attr("bucket: string")
  .Input("queue_handle: Ref(string)")
//  .Output("file_handle: string")
//  .Output("file_name: string")
  .SetIsStateful()
  .Doc(R"doc(
Fetches files from storage, synchronously reads them, and produces a Tensor[1,2]
with the container and shared name for the file.

queue_handle: a handle to the filename queue
file_handle: a Tensor(2) of strings to access the file resource in downstream nodes
file_name: a Tensor() of string for the unique key for this file
  )doc");

  class FileStorageOp : public OpKernel {
  public:
    FileStorageOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
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

      responseHandler.propertiesCallback = &FileStorageOp::responsePropertiesCallback;
      responseHandler.completeCallback = &FileStorageOp::responseCompleteCallback;

      getObjectHandler.responseHandler = FileStorageOp::responseHandler;
      getObjectHandler.getObjectDataCallback = &FileStorageOp::getObjectDataCallback;

      // Open connection
      S3_initialize("s3", S3_INIT_ALL, host.c_str());
    }

    void Compute(OpKernelContext* ctx) override {
      QueueInterface* queue;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "queue_handle", &queue));
      core::ScopedUnref unref_me(queue);

      OP_REQUIRES_OK(ctx, GetNextFilename(queue, &file_key, ctx));

      FILE* buffer = (FILE *) stdout; // Currently set to standard out
      S3_get_object(&bucketContext, file_key.c_str(), NULL, 0, 0, NULL, &getObjectHandler, buffer);
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
      FILE *outfile = (FILE *) callbackData;
      size_t wrote = fwrite(buffer, 1, bufferSize, outfile);
      return ((wrote < (size_t) bufferSize) ? S3StatusAbortedByCallback : S3StatusOK);
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

  REGISTER_KERNEL_BUILDER(Name("FileStorage").Device(DEVICE_CPU), FileStorageOp);

} // namespace tensorflow
