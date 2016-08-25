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
  .Output("num_records_out: int32")
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
        LOG(ERROR) << "Couldn't read the Ceph configuration file! error " << ret;
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

    ~CephWriterOp() {
      VLOG(DEBUG) << "Ceph writer " << this << " finishing\n";
      io_ctx.close();
      cluster.shutdown();
    }

    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      const Tensor *path, *column_t;
      start = clock();
      OP_REQUIRES_OK(ctx, ctx->input("file_name", &path));
      OP_REQUIRES_OK(ctx, ctx->input("column_handle", &column_t));
      auto filepath = path->scalar<string>()();
      auto column_vec = column_t->vec<string>();

      ResourceContainer<BufferList> *column;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(column_vec(0), column_vec(1), &column));
      core::ScopedUnref column_releaser(column);
      ResourceReleaser<BufferList> a(*column);
      tracepoint(bioflow, result_ready_queue_stop, column);

      auto &buffers = column->get()->get_when_ready();

      string full_path(filepath + record_suffix_);

      OP_REQUIRES_OK(ctx, WriteHeader(ctx, output_buf_));

      auto num_buffers = buffers.size();
      uint32_t num_records = header_.last_ordinal - header_.first_ordinal;
      uint32_t records_per_chunk = num_records / num_buffers;
      if (num_records % num_buffers != 0) {
        ++records_per_chunk;
      }

      decltype(num_records) i = 0, recs_per_chunk = records_per_chunk;
      if (compress_) {
        compress_buf_.clear();

        for (auto &buffer : buffers) {
          if (i + recs_per_chunk > num_records) {
            recs_per_chunk = num_records - i;
          }

          auto &data_buf = buffer.get(); // only need to do this on the first call
          OP_REQUIRES_OK(ctx, appendSegment(&data_buf[0], recs_per_chunk, compress_buf_, true));
          i += recs_per_chunk;
        }
        tracepoint(bioflow, total_align_stop, column);

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
      } else {
        for (auto &buffer : buffers) {
          if (i + recs_per_chunk > num_records) {
            recs_per_chunk = num_records - i;
          }

          auto &data_buf = buffer.get();

          OP_REQUIRES_OK(ctx, CephWriteColumn(full_path, &data_buf[0], recs_per_chunk));
          i += recs_per_chunk;
        }

        tracepoint(bioflow, total_align_stop, column);

        i = 0; recs_per_chunk = records_per_chunk;
        size_t expected_size;
        for (auto &buffer : buffers) {
          if (i + recs_per_chunk > num_records) {
            recs_per_chunk = num_records - i;
          }

          auto &data_buf = buffer.get();

          expected_size = data_buf.size() - recs_per_chunk;

          OP_REQUIRES_OK(ctx, CephWriteColumn(full_path, &data_buf[recs_per_chunk], expected_size));
          i += recs_per_chunk;
        }
      }
      Tensor *num_recs;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records_out", TensorShape({}), &num_recs));
      num_recs->scalar<int32>()() = num_records;
      tracepoint(bioflow, write_duration, full_path.c_str(), num_records);
    }

  private:
    string cluster_name;
    string user_name;
    string pool_name;
    string ceph_conf;
    librados::Rados cluster;
    librados::IoCtx io_ctx;
    vector<char> compress_buf_; // used to compress into
    vector<char> output_buf_; // used to compress into
    format::FileHeader header_;
    bool compress_ = false;
    string record_suffix_;
    clock_t start;

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
    Status CephWriteColumn(string& file_key, char* buf, size_t len)
    {
      int ret = 0;

      librados::bufferlist write_buf;
      write_buf.push_back(ceph::buffer::create_static(len, buf));

      // Create I/O Completion.
      /*librados::AioCompletion *write_completion = librados::Rados::aio_create_completion();
      ret = io_ctx.aio_write_full(file_key, write_completion, write_buf);
      if (ret < 0) {
        return Internal("Couldn't start read object! error: ", ret);
      }

      // Wait for the request to complete, and check that it succeeded.
      write_completion->wait_for_complete();
      ret = write_completion->get_return_value();
      if (ret < 0) {
        return Internal("Couldn't write object! error: ", ret);
      }*/
      ret = io_ctx.write_full(file_key, write_buf);
      if (ret < 0) {
        return Internal("Couldn't write object! error: ", ret);
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("CephWriter").Device(DEVICE_CPU), CephWriterOp);

} // namespace tensorflow {
