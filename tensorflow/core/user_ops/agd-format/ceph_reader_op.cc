#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "data.h"
#include "util.h"
#include "tensorflow/core/user_ops/agd-format/buffer.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <rados/librados.hpp>
#include <rados/buffer.h>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  REGISTER_OP("CephReader")
  .Attr("cluster_name: string")
  .Attr("user_name: string")
  .Attr("pool_name: string")
  .Attr("ceph_conf_path: string")
  .Attr("read_size: int64")
  .Input("buffer_handle: Ref(string)")
  .Input("queue_key: string")
  .Output("file_handle: string")
  .Output("file_name: string")
  .Doc(R"doc(
Obtains file names from a queue, fetches those files from Ceph storage using Librados,
and writes them to a buffer from a pool of buffers.

buffer_handle: a handle to the buffer pool
queue_key: key reference to the filename queue
file_handle: a Tensor(2) of strings to access the file resource in downstream nodes
file_name: a Tensor() of string for the unique key for this file
  )doc");

  class CephReaderOp : public OpKernel {
  public:
    CephReaderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("read_size", &read_size));

      int ret = 0;
      /* Initialize the cluster handle with the "ceph" cluster name and "client.admin" user */
      ret = cluster.init2(user_name.c_str(), cluster_name.c_str(), 0);
      if (ret < 0) {
              LOG(ERROR) << "Couldn't initialize the cluster handle! error " << ret;
              exit(EXIT_FAILURE);
      } else {
              VLOG(DEBUG) << "Created a cluster handle.";
      }

      /* Read a Ceph configuration file to configure the cluster handle. */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf));
      ret = cluster.conf_read_file(ceph_conf.c_str());
      if (ret < 0) {
          LOG(ERROR) << "Couldn't read the Ceph configuration file ('" << ceph_conf << "')! error " << ret;
              exit(EXIT_FAILURE);
      } else {
              VLOG(DEBUG) << "Read the Ceph configuration file.";
      }

      /* Connect to the cluster */
      ret = cluster.connect();
      if (ret < 0) {
              LOG(ERROR) << "Couldn't connect to cluster! error " << ret;
              exit(EXIT_FAILURE);
      } else {
              VLOG(DEBUG) << "Connected to the cluster.";
      }

      /* Set up IO context */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool_name));
      ret = cluster.ioctx_create(pool_name.c_str(), io_ctx);
      if (ret < 0) {
              LOG(ERROR) << "Couldn't set up ioctx! error " << ret;
              exit(EXIT_FAILURE);
      } else {
              VLOG(DEBUG) << "Created an ioctx for the pool.";
      }
    }

    ~CephReaderOp() {
      core::ScopedUnref unref_pool(ref_pool_);
      io_ctx.close();
      cluster.shutdown();
    }

    void Compute(OpKernelContext* ctx) override {
      if (!ref_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_handle", &ref_pool_));
      }
      start = clock();

      const Tensor *key_t;
      OP_REQUIRES_OK(ctx, ctx->input("queue_key", &key_t));
      string file_key = key_t->scalar<string>()();
      tracepoint(bioflow, process_key, file_key.c_str());

      ResourceContainer<Buffer> *rec_buffer;
      OP_REQUIRES_OK(ctx, ref_pool_->GetResource(&rec_buffer));
      rec_buffer->get()->reset();

      OP_REQUIRES_OK(ctx, CephReadObject(file_key.c_str(), rec_buffer));

      // Output tensors
      OP_REQUIRES_OK(ctx, rec_buffer->allocate_output("file_handle", ctx));

      Tensor *file_name;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_name", TensorShape({1}), &file_name));
      auto scalar = file_name->vec<string>();
      scalar(0) = file_key;
      tracepoint(bioflow, read_kernel, start, file_key.c_str(), rec_buffer->get()->size());
      tracepoint(bioflow, read_ready_queue_start, rec_buffer);
    }

  private:
    clock_t start;
    string cluster_name;
    string user_name;
    string pool_name;
    string ceph_conf;
    long long read_size;
    librados::Rados cluster;
    ReferencePool<Buffer> *ref_pool_ = nullptr;
    librados::IoCtx io_ctx;

    /* Read an object from Ceph synchronously */
    Status CephReadObject(const char* file_key, void *ref_buffer)
    {
      int ret = 0;
      auto buf = ((ResourceContainer<Buffer> *) ref_buffer)->get();

      size_t file_size;
      io_ctx.stat(file_key, &file_size, nullptr); // Get file size

      size_t data_read = 0;
      size_t read_len;
      size_t size_to_read = (size_t) read_size;
      buf->reserve(file_size);

      librados::bufferlist read_buf;
      while (data_read < file_size) {
        read_len = min(size_to_read, file_size - data_read);
        read_buf.push_back(ceph::buffer::create_static(read_len, &(*buf)[data_read]));

        // Create I/O Completion.
        /*librados::AioCompletion *read_completion = librados::Rados::aio_create_completion();
        ret = io_ctx.aio_read(file_key, read_completion, &read_buf, read_len, data_read);
        if (ret < 0) {
                LOG(INFO) << "Couldn't start read object! error " << ret;
                exit(EXIT_FAILURE);
        }
        data_read = data_read + read_len;

        // Wait for the request to complete, and check that it succeeded.
        read_completion->wait_for_complete();
        ret = read_completion->get_return_value();
        if (ret < 0) {
                LOG(INFO) << "Couldn't read object! error " << ret;
                exit(EXIT_FAILURE);
        }*/

        /* Test synchronous read */
        ret = io_ctx.read(file_key, read_buf, read_len, data_read);
        if (ret < 0) {
          return Internal("Couldn't call io_ctx.read. Received ", ret);
        }
        data_read += read_len;
        read_buf.clear();

        return Status::OK();
      }
    }
  };

  REGISTER_KERNEL_BUILDER(Name("CephReader").Device(DEVICE_CPU), CephReaderOp);

} // namespace tensorflow {
