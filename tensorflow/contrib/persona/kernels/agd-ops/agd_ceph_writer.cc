
#include "agd_ceph_writer.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  AGDCephWriterBase::AGDCephWriterBase(OpKernelConstruction *ctx) : AGDWriterBase(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf_path_));

    int ret = 0;
    ret = cluster_.init2(user_name_.c_str(), cluster_name_.c_str(), 0);
    OP_REQUIRES(ctx, ret == 0, Internal("cluster.init2 returned ", ret, " with user ", user_name_,
                                        " and cluster ", cluster_name_));
    ret = cluster_.conf_read_file(ceph_conf_path_.c_str());
    OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster::conf_read_file for file ", ceph_conf_path_, " returned error code ", ret));

    ret = cluster_.connect();
    OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster::connect returned error status ", ret));
  }

  void AGDCephWriterBase::Compute(OpKernelContext *ctx) {
    OP_REQUIRES_OK(ctx, SetPool(ctx));

    const Tensor *key_t, *resource_t;
    OP_REQUIRES_OK(ctx, ctx->input("path", &key_t));
    OP_REQUIRES_OK(ctx, ctx->input("resource_handle", &resource_t));
    auto key = key_t->scalar<string>()() + record_suffix_;
    auto resource_vec = resource_t->vec<string>();

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

  Status AGDCephWriterBase::SetPool(OpKernelContext *ctx) {
    const Tensor *pool_name_t;
    TF_RETURN_IF_ERROR(ctx->input("pool_name", &pool_name_t));
    auto &pool_name = pool_name_t->scalar<string>()();
    if (pool_name_ != pool_name) {
      pool_name_ = pool_name;
      int ret = cluster_.ioctx_create(pool_name_.c_str(), io_ctx_);
      if (ret != 0) {
        return Internal("Ceph cluster ioctx_create for pool ", pool_name_, " return error ", ret);
      }
    }
    return Status::OK();
  }
} // namespace tensorflow {

