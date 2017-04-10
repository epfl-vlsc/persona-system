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
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("read_size", &read_size));

      int ret = 0;
      /* Initialize the cluster handle with the "ceph" cluster name and "client.admin" user */
      ret = cluster.init2(user_name.c_str(), cluster_name.c_str(), 0);
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster init2\nUsername: ", user_name, "\nCluster Name: ", cluster_name, "\nReturn code: ", ret));

      /* Read a Ceph configuration file to configure the cluster handle. */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf));
      ret = cluster.conf_read_file(ceph_conf.c_str());
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph conf file at '", ceph_conf, "' returned ", ret, " when attempting to open"));

      /* Connect to the cluster */
      ret = cluster.connect();
      OP_REQUIRES(ctx, ret == 0, Internal("Cluster connect returned: ", ret));
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

      const Tensor *key_t, *pool_name_t;
      OP_REQUIRES_OK(ctx, ctx->input("key", &key_t));
      OP_REQUIRES_OK(ctx, ctx->input("pool_name", &pool_name_t));
      auto file_key = key_t->scalar<string>()();
      auto pool_name = pool_name_t->scalar<string>()();
      tracepoint(bioflow, process_key, file_key.c_str());

      auto start = chrono::high_resolution_clock::now();

      ResourceContainer<Buffer> *rec_buffer;
      OP_REQUIRES_OK(ctx, ref_pool_->GetResource(&rec_buffer));
      rec_buffer->get()->reset();

      auto read_only_start = chrono::high_resolution_clock::now();
      OP_REQUIRES_OK(ctx, CephReadObject(file_key, pool_name, rec_buffer));
      auto duration = TRACEPOINT_DURATION_CALC(read_only_start);
      tracepoint(bioflow, ceph_read, duration);

      // Output tensors
      OP_REQUIRES_OK(ctx, rec_buffer->allocate_output("file_handle", ctx));

      duration = TRACEPOINT_DURATION_CALC(start);
      tracepoint(bioflow, chunk_read, file_key.c_str(), duration);
    }

  private:
    librados::IoCtx io_ctx;
    string cluster_name;
    string user_name;
    string ceph_conf;
    long long read_size;
    librados::Rados cluster;
    ReferencePool<Buffer> *ref_pool_ = nullptr;

    /* Read an object from Ceph synchronously */
    Status CephReadObject(const string &file_key, const string &pool_name, ResourceContainer<Buffer> *ref_buffer) {
      int ret = 0;
      if (pool_name != io_ctx.get_pool_name()) {
        // TODO need to close before creating a new one?
        ret = cluster.ioctx_create(pool_name.c_str(), io_ctx);
        if (ret != 0) {
          return Internal("CephReader: Unable to create new io ctx for new pool ", pool_name, ". Got return code ", ret);
        } else {
          VLOG(INFO) << "Creating a new context because pool name changed to " << pool_name;
        }
      }
      auto buf = ref_buffer->get();

      size_t file_size;
      time_t pmtime;
      ret = io_ctx.stat(file_key, &file_size, &pmtime); // Get file size
      if (ret != 0) {
        return Internal("CephReader: io_ctx.stat() return ", ret, " for key ", file_key);
      }
      /*char bufff[32];
      struct tm* tt = localtime(&pmtime);
      strftime(bufff, sizeof(bufff), "%b %d %H:%M", tt);
      LOG(INFO) << "Object " << file_key << " was last modified " << bufff;*/

      size_t data_read = 0;
      size_t read_len;
      size_t size_to_read = (size_t) read_size;
      buf->resize(file_size);

      librados::bufferlist read_buf;
      while (data_read < file_size) {
        read_len = min(size_to_read, file_size - data_read);
        read_buf.push_back(ceph::buffer::create_static(read_len, &(*buf)[data_read]));

        // Create I/O Completion.
        librados::AioCompletion *read_completion = librados::Rados::aio_create_completion();
        ret = io_ctx.aio_read(file_key, read_completion, &read_buf, read_len, data_read);
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

      VLOG(INFO) << "reader read " << data_read << " bytes from ceph object " << file_key;
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("CephReader").Device(DEVICE_CPU), CephReaderOp);

} // namespace tensorflow {
