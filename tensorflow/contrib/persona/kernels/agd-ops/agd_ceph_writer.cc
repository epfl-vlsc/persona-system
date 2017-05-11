
#include "agd_ceph_writer.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  AGDCephWriterBase::AGDCephWriterBase(OpKernelConstruction *ctx) : AGDWriterBase(ctx) {
    string cluster_name, user_name, pool_name, ceph_conf_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf_path));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool_name));

    int ret = 0;
    ret = cluster_.init2(user_name.c_str(), cluster_name.c_str(), 0);
    OP_REQUIRES(ctx, ret == 0, Internal("cluster.init2 returned ", ret, " with user ", user_name,
                                        " and cluster ", cluster_name));
    ret = cluster_.conf_read_file(ceph_conf_path.c_str());
    OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster::conf_read_file for file ", ceph_conf_path, " returned error code ", ret));

    ret = cluster_.connect();
    OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster::connect returned error status ", ret));

    ret = cluster_.ioctx_create(pool_name.c_str(), io_ctx_);
    OP_REQUIRES(ctx, ret == 0, Internal("Ceph Writer: unable to open pool with pool name: ", pool_name));
  }

  void AGDCephWriterBase::Compute(OpKernelContext *ctx) {
    const Tensor *key_t, *resource_t, *namespace_t;
    OP_REQUIRES_OK(ctx, ctx->input("path", &key_t));
    OP_REQUIRES_OK(ctx, ctx->input("resource_handle", &resource_t));
    OP_REQUIRES_OK(ctx, ctx->input("namespace", &namespace_t));
    auto &name_space = namespace_t->scalar<string>()();
    auto &key = key_t->scalar<string>()();
    auto resource_vec = resource_t->vec<string>();

    io_ctx_.set_namespace(name_space);

    OP_REQUIRES_OK(ctx, SetHeaderValues(ctx));

    write_buf_list_.clear();

    OP_REQUIRES_OK(ctx, WritePayload(ctx, resource_vec(0), resource_vec(1), key, write_buf_list_));

    OP_REQUIRES_OK(ctx, SetOutputKey(ctx, key));
  }

  Status AGDCephWriterBase::SendWrite(librados::bufferlist &write_buf_list, const string &key) {
    write_buf_list_.push_front(ceph::buffer::create_static(sizeof(header_),
                                                           reinterpret_cast<char*>(&header_)));

    // TODO max default write size is 90M. We should have it chunk this up if necessary
    auto ret = io_ctx_.write_full(key, write_buf_list_);
    if (ret != 0) {
      return Internal("Ceph write failed with code ", ret, " with size ", write_buf_list_.length());
    } else {
      return Status::OK();
    }
  }
} // namespace tensorflow {

