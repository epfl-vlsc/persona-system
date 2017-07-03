#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/data.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include <list>
#include <rados/librados.hpp>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class CephReaderOp : public OpKernel {
  public:
    CephReaderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      string pool, user_name, cluster_name, ceph_conf;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("read_size", &read_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool));

      int ret = 0;
      /* Initialize the cluster handle with the "ceph" cluster name and "client.admin" user */
      ret = cluster_.init2(user_name.c_str(), cluster_name.c_str(), 0);
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster init2\nUsername: ", user_name, "\nCluster Name: ", cluster_name, "\nReturn code: ", ret));

      /* Read a Ceph configuration file to configure the cluster handle. */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf));
      ret = cluster_.conf_read_file(ceph_conf.c_str());
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph conf file at '", ceph_conf, "' returned ", ret, " when attempting to open"));

      /* Connect to the cluster */
      ret = cluster_.connect();
      OP_REQUIRES(ctx, ret == 0, Internal("Cluster connect returned: ", ret));

      ret = cluster_.ioctx_create(pool.c_str(), io_ctx_);
      OP_REQUIRES(ctx, ret == 0, Internal("Unable to create contetx for pool: ", pool));
    }

    ~CephReaderOp() {
      core::ScopedUnref unref_pool(ref_pool_);
      io_ctx_.close();
      cluster_.shutdown();
    }

    void Compute(OpKernelContext* ctx) override {
      if (!ref_pool_) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "buffer_pool", &ref_pool_));
      }

      const Tensor *key_t, *namespace_t;
      OP_REQUIRES_OK(ctx, ctx->input("key", &key_t));
      OP_REQUIRES_OK(ctx, ctx->input("namespace", &namespace_t));
      auto file_key = key_t->scalar<string>()();
      auto name_space = namespace_t->scalar<string>()();

      ResourceContainer<Buffer> *rec_buffer;
      OP_REQUIRES_OK(ctx, ref_pool_->GetResource(&rec_buffer));
      rec_buffer->get()->reset();

      OP_REQUIRES_OK(ctx, CephReadObject(file_key, name_space, rec_buffer));

      // Output tensors
      OP_REQUIRES_OK(ctx, rec_buffer->allocate_output("file_handle", ctx));
    }

  private:
    librados::IoCtx io_ctx_;
    long long read_size_;
    librados::Rados cluster_;
    ReferencePool<Buffer> *ref_pool_ = nullptr;

    /* Read an object from Ceph synchronously */
    Status CephReadObject(const string &file_key, const string &name_space, ResourceContainer<Buffer> *ref_buffer) {
      int ret;
      auto buf = ref_buffer->get();
      io_ctx_.set_namespace(name_space);

      size_t file_size;
      time_t pmtime;
      ret = io_ctx_.stat(file_key, &file_size, &pmtime); // Get file size
      if (ret != 0) {
        return Internal("CephReader: io_ctx.stat() return ", ret, " for key ", file_key);
      }
      /*char bufff[32];
      struct tm* tt = localtime(&pmtime);
      strftime(bufff, sizeof(bufff), "%b %d %H:%M", tt);
      LOG(INFO) << "Object " << file_key << " was last modified " << bufff;*/

      size_t data_read = 0;
      size_t read_len;
      size_t size_to_read = (size_t) read_size_;
      buf->resize(file_size);

      librados::bufferlist read_buf;
      while (data_read < file_size) {
        read_len = min(size_to_read, file_size - data_read);
        read_buf.push_back(ceph::buffer::create_static(read_len, &(*buf)[data_read]));

        // Create I/O Completion.
        librados::AioCompletion *read_completion = librados::Rados::aio_create_completion();
        ret = io_ctx_.aio_read(file_key, read_completion, &read_buf, read_len, data_read);
        if (ret < 0) {
          return Internal("Ceph Reader: unable to start read object. Received error ", ret);
        }
        data_read = data_read + read_len;

        // Wait for the request to complete, and check that it succeeded.
        read_completion->wait_for_complete();
        ret = read_completion->get_return_value();
        if (ret < 0) {
          return Internal("Ceph Reader: unable to read object. Got error ", ret);
        }
        read_buf.clear();
        read_completion->release();

        /* Test synchronous read */
        /*ret = io_ctx.read(file_key, read_buf, read_len, data_read);
        if (ret < 0) {
          return Internal("Couldn't call io_ctx.read. Received ", ret);
        }
        data_read += read_len;
        read_buf.clear();*/
      }

      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("CephReader").Device(DEVICE_CPU), CephReaderOp);

} // namespace tensorflow {
