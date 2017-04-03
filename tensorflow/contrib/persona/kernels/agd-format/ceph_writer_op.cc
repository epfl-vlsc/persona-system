#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "compression.h"
#include "buffer_list.h"
#include "format.h"
#include "data.h"
#include "util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <list>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <rados/librados.hpp>
#include <rados/buffer.h>

namespace tensorflow {
  using namespace errors;
  using namespace std;


  class CephWriterOp : public OpKernel {
  public:
    CephWriterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

      // file format init
      using namespace format;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compress", &compress_));
      OP_REQUIRES(ctx, !compress_, Internal("op doesn't support compression yet"));
      string s;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_id", &s));
      auto max_size = sizeof(header_.string_id);
      OP_REQUIRES(ctx, s.length() < max_size,
                  Internal("record_id for column header '", s, "' greater than 32 characters"));
      strncpy(header_.string_id, s.c_str(), max_size);

      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_type", &s));
      RecordType t;
      if (s.compare("base") == 0) {
        t = RecordType::BASES;
      } else if (s.compare("qual") == 0) {
        t = RecordType::QUALITIES;
      } else if (s.compare("meta") == 0) {
        t = RecordType::COMMENTS;
      } else { // no need to check. we're saved by string enum types if TF
        t = RecordType::ALIGNMENT;
      }
      record_suffix_ = "." + s;
      header_.record_type = static_cast<uint8_t>(t);
      header_.compression_type = format::CompressionType::UNCOMPRESSED;

      // ceph cluster_ init
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name_", &cluster_name_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name_", &user_name_));

      int ret = 0;
      /* Initialize the cluster_ handle with the "ceph" cluster_ name and "client.admin" user */
      ret = cluster_.init2(user_name_.c_str(), cluster_name_.c_str(), 0);
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph cluster_ init2\nUsername: ", user_name_, "\nCluster Name: ", cluster_name_, "\nReturn code: ", ret));

      /* Read a Ceph configuration file to configure the cluster_ handle. */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("ceph_conf_path", &ceph_conf_));
      ret = cluster_.conf_read_file(ceph_conf_.c_str());
      OP_REQUIRES(ctx, ret == 0, Internal("Ceph conf file at '", ceph_conf_, "' returned ", ret, " when attempting to open"));

      /* Connect to the cluster_ */
      ret = cluster_.connect();
      OP_REQUIRES(ctx, ret == 0, Internal("Cluster connect returned: ", ret));
    }

    ~CephWriterOp() {
      VLOG(INFO) << "Ceph writer " << this << " finishing\n";
      //io_ctx.watch_flush();
      io_ctx.close();
      cluster_.shutdown();
    }

    void Compute(OpKernelContext* ctx) override {
      auto start = chrono::high_resolution_clock::now();
      const Tensor *path, *column_t, *pool_name_t, *record_id_t;
      int ret;

      OP_REQUIRES_OK(ctx, ctx->input("file_name", &path));
      OP_REQUIRES_OK(ctx, ctx->input("column_handle", &column_t));
      OP_REQUIRES_OK(ctx, ctx->input("pool_name", &pool_name_t));
      OP_REQUIRES_OK(ctx, ctx->input("record_id", &record_id_t));
      auto filepath = path->scalar<string>()();
      auto column_vec = column_t->vec<string>();
      auto pool_name = pool_name_t->scalar<string>()();
      auto record_id = record_id_t->scalar<string>()();

      if (pool_name != io_ctx.get_pool_name()) {
        ret = cluster_.ioctx_create(pool_name.c_str(), io_ctx);
        OP_REQUIRES(ctx, ret == 0, Internal("ceph writer couldn't set up ioctx! error code: ", ret));
      }

      if (strncmp(record_id.c_str(), &header_.string_id[0], sizeof(header_.string_id)) != 0) {
        strncpy(&header_.string_id[0], record_id.c_str(), sizeof(header_.string_id));
      }

      ResourceContainer<BufferList> *column;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(column_vec(0), column_vec(1), &column));
      core::ScopedUnref column_releaser(column);
      ResourceReleaser<BufferList> a(*column);

      auto *buf_list = column->get();
      buf_list->wait_for_ready();

      string full_path(filepath + record_suffix_);
      OP_REQUIRES_OK(ctx, PrepHeader(ctx));

      auto num_buffers = buf_list->size();
      size_t i;
      index_.reset();
      payload_.reset();

      // We currently check to make sure compression output is off in the constructor
      for (i = 0; i < num_buffers; ++i) {
        auto &index = (*buf_list)[i].index();
        //OP_REQUIRES(ctx, s1 != 0, Internal("ceph write got empty record!"));
        OP_REQUIRES_OK(ctx, index_.AppendBuffer(&index[0], index.size()));
      }

      for (i = 0; i < num_buffers; ++i) {
        auto &data = (*buf_list)[i].data();
        OP_REQUIRES_OK(ctx, payload_.AppendBuffer(&data[0], data.size()));
      }

      auto write_only_start = chrono::high_resolution_clock::now();
      write_buf.clear();
      write_buf.push_back(ceph::buffer::create_static(sizeof(header_), reinterpret_cast<char*>(&header_)));
      write_buf.push_back(ceph::buffer::create_static(index_.size(), const_cast<char*>(index_.data())));
      write_buf.push_back(ceph::buffer::create_static(payload_.size(), const_cast<char*>(payload_.data())));

      ret = io_ctx.write_full(full_path, write_buf);
      OP_REQUIRES(ctx, ret >= 0, Internal("Couldn't write object! error: ", ret));
      auto duration = TRACEPOINT_DURATION_CALC(write_only_start);
      tracepoint(bioflow, ceph_write, duration);

      duration = TRACEPOINT_DURATION_CALC(start);
      tracepoint(bioflow, chunk_write, filepath.c_str(), duration);

      Tensor *num_recs;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("key_out", TensorShape({}), &num_recs));
      num_recs->scalar<string>()() = filepath;
    }

  private:
    librados::bufferlist write_buf;
    string cluster_name_, user_name_, ceph_conf_, record_suffix_;
    Buffer index_, payload_;
    librados::Rados cluster_;
    librados::IoCtx io_ctx;
    format::FileHeader header_;
    bool compress_ = false;

    Status PrepHeader(OpKernelContext *ctx) {
      const Tensor *tensor;
      uint64_t tmp64;
      TF_RETURN_IF_ERROR(ctx->input("first_ordinal", &tensor));
      tmp64 = static_cast<decltype(tmp64)>(tensor->scalar<int64>()());
      header_.first_ordinal = tmp64;

      TF_RETURN_IF_ERROR(ctx->input("num_records", &tensor));
      tmp64 = static_cast<decltype(tmp64)>(tensor->scalar<int32>()());
      header_.last_ordinal = header_.first_ordinal + tmp64;

      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("CephWriter").Device(DEVICE_CPU), CephWriterOp);

} // namespace tensorflow {
