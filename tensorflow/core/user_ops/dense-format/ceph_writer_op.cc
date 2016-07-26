#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "data.h"
#include "tensorflow/core/user_ops/dense-format/buffer.h"
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <rados/librados.hpp>
#include <rados/buffer.h>

namespace tensorflow {
  using namespace std;

  REGISTER_OP("CephWriter")
  .Attr("cluster_name: string")
  .Attr("user_name: string")
  .Attr("pool_name: string")
  .Attr("ceph_conf_path: string")
  .Input("column_handle: string")
  .Input("file_name: string")
  .Doc(R"doc(
Writes data in column_handle to object file_name in specified Ceph cluster.

column_handle: a handle to the buffer pool
file_name: a Tensor() of string for the unique key for this file
  )doc");

  class CephWriterOp : public OpKernel {
  public:
    CephWriterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));

      int ret = 0;
      /* Initialize the cluster handle with the "ceph" cluster name and "client.admin" user */
      ret = cluster.init2(user_name.c_str(), cluster_name.c_str(), 0);
      if (ret < 0) {
              LOG(INFO) << "Couldn't initialize the cluster handle! error " << ret;
              exit(EXIT_FAILURE);
      } else {
              LOG(INFO) << "Created a cluster handle.";
      }

      /* Read a Ceph configuration file to configure the cluster handle. */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf));
      ret = cluster.conf_read_file(ceph_conf.c_str());
      if (ret < 0) {
              LOG(INFO) << "Couldn't read the Ceph configuration file! error " << ret;
              exit(EXIT_FAILURE);
      } else {
              LOG(INFO) << "Read the Ceph configuration file.";
      }

      /* Connect to the cluster */
      ret = cluster.connect();
      if (ret < 0) {
              LOG(INFO) << "Couldn't connect to cluster! error " << ret;
              exit(EXIT_FAILURE);
      } else {
              LOG(INFO) << "Connected to the cluster.";
      }

      /* Set up IO context */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool_name));
      ret = cluster.ioctx_create(pool_name.c_str(), io_ctx);
      if (ret < 0) {
              LOG(INFO) << "Couldn't set up ioctx! error " << ret;
              exit(EXIT_FAILURE);
      } else {
              LOG(INFO) << "Created an ioctx for the pool.";
      }
    }

    ~CephWriterOp() {
      core::ScopedUnref unref_pool(ref_pool_);
      io_ctx.close();
      cluster.shutdown();
    }

    void Compute(OpKernelContext* ctx) override {

      const Tensor *key_t, *column_t;
      OP_REQUIRES_OK(ctx, ctx->input("file_name", &key_t));
      string file_key = key_t->scalar<string>()();
      OP_REQUIRES_OK(ctx, ctx->input("column_handle", &column_t));
      auto column_vec = column_t->vec<string>();

      ResourceContainer<Data> *column;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(column_vec(0), column_vec(1), &column));
      
      CephWriteColumn(file_key, column->get());

      core::ScopedUnref a(column);
      {
        ResourceReleaser<Data> b(*column); // make sure destructs first
      }
    }

  private:
    string cluster_name;
    string user_name;
    string pool_name;
    string ceph_conf;
    long long read_size;
    librados::Rados cluster;
    ReferencePool<Buffer> *ref_pool_;
    librados::IoCtx io_ctx;

    /* Read an object from Ceph asynchronously */
    void CephWriteColumn(string& file_key, Data* column)
    {
      int ret = 0;

      size_t write_size = column->size();
      LOG(INFO) << "Size of write is " << write_size;

      librados::bufferlist write_buf;
      write_buf.push_back(ceph::buffer::create_static(write_size, (char*)column->data()));

      // Create I/O Completion.
      librados::AioCompletion *write_completion = librados::Rados::aio_create_completion();
      ret = io_ctx.aio_write_full(file_key, write_completion, write_buf);
      if (ret < 0) {
              LOG(INFO) << "Couldn't start read object! error " << ret;
              exit(EXIT_FAILURE);
      }

      // Wait for the request to complete, and check that it succeeded.
      write_completion->wait_for_complete();
      ret = write_completion->get_return_value();
      if (ret < 0) {
              LOG(INFO) << "Couldn't write object! error " << ret;
              exit(EXIT_FAILURE);
      }
      LOG(INFO) << "wrote object asynchronously.";

    }
  };

  REGISTER_KERNEL_BUILDER(Name("CephWriter").Device(DEVICE_CPU), CephWriterOp);

} // namespace tensorflow {
