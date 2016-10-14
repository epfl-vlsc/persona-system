#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"
#include "compression.h"
#include "buffer_list.h"
#include "format.h"
#include "data.h"
#include "util.h"
#include "tensorflow/core/user_ops/agd-format/buffer.h"
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <rados/librados.hpp>
#include <rados/buffer.h>

namespace tensorflow {
  using namespace errors;
  using namespace std;

  REGISTER_OP("CephWriter")
  .Attr("cluster_name: string")
  .Attr("user_name: string")
  .Attr("pool_name: string")
  .Attr("ceph_conf_path: string")
  .Attr("compress: bool")
  .Attr("record_id: string")
  .Attr("record_type: {'base', 'qual', 'meta', 'results'}")
  .Input("column_handle: string")
  .Input("file_name: string")
  .Input("first_ordinal: int64")
  .Input("num_records: int32")
  .Output("key_out: string")
  .Doc(R"doc(
Writes data in column_handle to object file_name in specified Ceph cluster.

column_handle: a handle to the buffer pool
file_name: a Tensor() of string for the unique key for this file
compress: whether or not to compress the column
  )doc");

  class CephWriterOp : public OpKernel {
  public:
    CephWriterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

      // file format init
      using namespace format;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compress", &compress_));
      string s;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_id", &s));
      auto max_size = sizeof(header_.string_id);
      OP_REQUIRES(ctx, s.length() < max_size,
                  errors::Internal("record_id for column header '", s, "' greater than 32 characters"));
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

      // ceph cluster init
      OP_REQUIRES_OK(ctx, ctx->GetAttr("cluster_name", &cluster_name));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("user_name", &user_name));

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

      /* Set up IO context */
      OP_REQUIRES_OK(ctx, ctx->GetAttr("pool_name", &pool_name));
      ret = cluster.ioctx_create(pool_name.c_str(), io_ctx);
      OP_REQUIRES(ctx, ret == 0, Internal("ceph writer couldn't set up ioctx! error code: ", ret));
    }

    ~CephWriterOp() {
      VLOG(DEBUG) << "Ceph writer " << this << " finishing\n";
      //io_ctx.watch_flush();
      io_ctx.close();
      cluster.shutdown();
    }

    void Compute(OpKernelContext* ctx) override {
      auto start = chrono::high_resolution_clock::now();
      const Tensor *path, *column_t;
      OP_REQUIRES_OK(ctx, ctx->input("file_name", &path));
      OP_REQUIRES_OK(ctx, ctx->input("column_handle", &column_t));
      auto filepath = path->scalar<string>()();
      auto column_vec = column_t->vec<string>();

      ResourceContainer<BufferList> *column;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(column_vec(0), column_vec(1), &column));
      core::ScopedUnref column_releaser(column);
      ResourceReleaser<BufferList> a(*column);

      auto *buf_list = column->get();
      buf_list->wait_for_ready();

      string full_path(filepath + record_suffix_);

      OP_REQUIRES_OK(ctx, WriteHeader(ctx, output_buf_));

      auto num_buffers = buf_list->size();
      size_t i;
      uint32_t num_records = header_.last_ordinal - header_.first_ordinal;
      index_.reset();
      payload_.reset();

      if (compress_) {
        OP_REQUIRES(ctx, false, Internal("Compression not supported for ceph writers"));
#if 0
        compress_buf_.clear();

        for (auto &buffer : buffers) {
          if (i + recs_per_chunk > num_records) {
            recs_per_chunk = num_records - i;
          }

          auto &data_buf = buffer.get(); // only need to do this on the first call
          OP_REQUIRES_OK(ctx, appendSegment(&data_buf[0], recs_per_chunk, compress_buf_, true));
          i += recs_per_chunk;
        }

        i = 0; recs_per_chunk = records_per_chunk;
        size_t expected_size;
        for (auto &buffer : buffers) {
          if (i + recs_per_chunk > num_records) {
            recs_per_chunk = num_records - i;
          }

          auto &data_buf = buffer.get();

          expected_size = data_buf.size() - recs_per_chunk;
          OP_REQUIRES_OK(ctx, appendSegment(&data_buf[recs_per_chunk], expected_size, compress_buf_, true));
          i += recs_per_chunk;
        }

        output_buf_.clear();
        OP_REQUIRES_OK(ctx, compressGZIP(&compress_buf_[0], compress_buf_.size(), output_buf_));
        OP_REQUIRES_OK(ctx, CephWriteColumn(full_path, &output_buf_[0], output_buf_.size()));
#endif
      } else {
        for (i = 0; i < num_buffers; ++i) {
          auto &index = (*buf_list)[i].index();
          //OP_REQUIRES(ctx, s1 != 0, Internal("ceph write got empty record!"));
          OP_REQUIRES_OK(ctx, index_.AppendBuffer(&index[0], index.size()));
        }

        for (i = 0; i < num_buffers; ++i) {
          auto &data = (*buf_list)[i].data();
          OP_REQUIRES_OK(ctx, payload_.AppendBuffer(&data[0], data.size()));
        }
      }

      auto write_only_start = chrono::high_resolution_clock::now();
      write_buf.clear();
      write_buf.push_back(ceph::buffer::create_static(index_.size(), const_cast<char*>(index_.data())));
      write_buf.push_back(ceph::buffer::create_static(payload_.size(), const_cast<char*>(payload_.data())));

      auto ret = io_ctx.write_full(full_path, write_buf);
      OP_REQUIRES(ctx, ret >= 0, Internal("Couldn't write object! error: ", ret));
      auto duration = TRACEPOINT_DURATION_CALC(write_only_start);
      LOG(DEBUG) << "Write duration: " << duration.count();
      tracepoint(bioflow, ceph_write, duration);

      duration = TRACEPOINT_DURATION_CALC(start);
      tracepoint(bioflow, chunk_write, filepath.c_str(), duration);

      Tensor *num_recs;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("key_out", TensorShape({}), &num_recs));
      num_recs->scalar<string>()() = filepath;
    }

  private:
    string cluster_name;
    string user_name;
    string pool_name;
    string ceph_conf;
    Buffer index_, payload_;
    librados::Rados cluster;
    librados::IoCtx io_ctx;
    vector<char> compress_buf_; // used to compress into
    vector<char> output_buf_; // used to compress into
    format::FileHeader header_;
    bool compress_ = false;
    string record_suffix_;

    Status WriteHeader(OpKernelContext *ctx, vector<char>& buf) {
      const Tensor *tensor;
      uint64_t tmp64;
      TF_RETURN_IF_ERROR(ctx->input("first_ordinal", &tensor));
      tmp64 = static_cast<decltype(tmp64)>(tensor->scalar<int64>()());
      header_.first_ordinal = tmp64;

      TF_RETURN_IF_ERROR(ctx->input("num_records", &tensor));
      tmp64 = static_cast<decltype(tmp64)>(tensor->scalar<int32>()());
      header_.last_ordinal = header_.first_ordinal + tmp64;

      appendSegment(reinterpret_cast<const char*>(&header_), sizeof(header_), buf, false);
      return Status::OK();
    }

    /* Write an object to Ceph synchronously */
    librados::bufferlist write_buf;
    Status CephWriteColumn(string& file_key, char* buf, size_t len)
    {
      int ret = 0;
      write_buf.push_back(ceph::buffer::create_static(len, buf));

      // Create I/O Completion.
      librados::AioCompletion *write_completion = librados::Rados::aio_create_completion();
      ret = io_ctx.aio_write_full(file_key, write_completion, write_buf);
      if (ret < 0) {
        return Internal("Couldn't start read object! error: ", ret);
      }

      // Wait for the request to complete, and check that it succeeded.
      write_completion->wait_for_complete();
      ret = write_completion->get_return_value();
      if (ret < 0) {
        return Internal("Couldn't write object! error: ", ret);
      }
      /*ret = io_ctx.write_full(file_key, write_buf);
      if (ret < 0) {
        return Internal("Couldn't write object! error: ", ret);
      }*/
      write_buf.clear();
      write_completion->release();
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("CephWriter").Device(DEVICE_CPU), CephWriterOp);

} // namespace tensorflow {
