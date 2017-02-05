#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "data.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include "compression.h"
#include "buffer_list.h"
#include "format.h"
#include "util.h"
#include <rados/librados.hpp>
#include <rados/buffer.h>

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("AGDCephWriteColumns");
  }

  class AGDCephWriteColumnsOp : public OpKernel {
  public:
    AGDCephWriteColumnsOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      using namespace format;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("compress", &compress_));
      string rec_id;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_id", &rec_id));
      auto max_size = sizeof(format::FileHeader::string_id);
      OP_REQUIRES(ctx, rec_id.length() < max_size,
                  Internal("record_id for column header '", rec_id, "' greater than 32 characters"));

      vector<string> rec_types;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_type", &rec_types));
      for (size_t i = 0; i < rec_types.size(); ++i) {
        auto& t_in = rec_types[i];
        format::FileHeader header;
        strncpy(header.string_id, rec_id.c_str(), max_size);

        RecordType t;
        if (t_in.compare("base") == 0) {
          t = RecordType::BASES;
        } else if (t_in.compare("qual") == 0) {
          t = RecordType::QUALITIES;
        } else if (t_in.compare("metadata") == 0) {
          t = RecordType::COMMENTS;
        } else { // no need to check. we're saved by string enum types if TF
          t = RecordType::ALIGNMENT;
        }
        record_suffixes_.push_back("." + t_in);
        header.record_type = static_cast<uint8_t>(t);
        header.compression_type = compress_ ? CompressionType::GZIP : CompressionType::UNCOMPRESSED;
        headers_.push_back(header);
      }
      
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
      LOG(INFO) << "poolname: " << pool_name;
      ret = cluster.ioctx_create(pool_name.c_str(), io_ctx);
      OP_REQUIRES(ctx, ret == 0, Internal("ceph writer couldn't set up ioctx! error code: ", ret));
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor *path, *column_t;
      OP_REQUIRES_OK(ctx, ctx->input("file_path", &path));
      OP_REQUIRES_OK(ctx, ctx->input("column_handle", &column_t));
      auto filepath = path->scalar<string>()();
      auto column_vec = column_t->vec<string>();

      OP_REQUIRES_OK(ctx, InitHeaders(ctx));

      ResourceContainer<BufferList> *columns;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(column_vec(0), column_vec(1), &columns));
      core::ScopedUnref column_releaser(columns);
      ResourceReleaser<BufferList> a(*columns);

      auto *buf_list = columns->get();
      auto num_buffers = buf_list->size();

      // do this after the wait, so we are only timing the write, and NOT part of the alignment
      //start = chrono::high_resolution_clock::now();
      
      for (size_t i = 0; i < num_buffers; ++i) {

        write_buf_.clear();
        string full_path(filepath + record_suffixes_[i]);
        LOG(INFO) << "writing file to ceph " << full_path;

        write_buf_.push_back(ceph::buffer::create_static(sizeof(format::FileHeader), (char*)(&headers_[i])));

        if (compress_) {
          OP_REQUIRES(ctx, false, Internal("Compressed out writing for columns not yet supported"));
        } else {
          auto &index = (*buf_list)[i].index();
          auto s1 = index.size();
          if (s1 == 0) {
            OP_REQUIRES(ctx, s1 != 0,
                        Internal("Parallel column writer got an empty index entry (size 0)!"));
          }
          write_buf_.push_back(ceph::buffer::create_static(s1, const_cast<char*>(&index[0])));
          
          auto ret = io_ctx.write_full(full_path, write_buf_);
          OP_REQUIRES(ctx, ret >= 0, Internal("Couldn't write object! error: ", ret));

          auto &data = (*buf_list)[i].data();
          size_t bytes_to_append = data.size();
          size_t offset = 0;
          // append in pieces, because there is a max write size
          while (bytes_to_append > 0) {
            // 90M is the default max write size in Rados
            write_buf_.clear();
            auto size = (bytes_to_append > 90000000) ? 90000000 : bytes_to_append;
            write_buf_.push_back(ceph::buffer::create_static(size, const_cast<char*>(&data[offset])));
            auto ret = io_ctx.append(full_path, write_buf_, size);
            OP_REQUIRES(ctx, ret >= 0, Internal("Couldn't write object! error: ", ret));
            bytes_to_append -= size;
            offset += size;
          }
        }
      }


      Tensor *key_out;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("key_out", TensorShape({}), &key_out));
      key_out->scalar<string>()() = filepath;

      //auto duration = TRACEPOINT_DURATION_CALC(start);
      //tracepoint(bioflow, chunk_read, filepath.c_str(), duration);
    }

  private:

    Status InitHeaders(OpKernelContext *ctx) {
      uint64_t first_ord, last_ord;
      const Tensor *tensor;
      TF_RETURN_IF_ERROR(ctx->input("first_ordinal", &tensor));
      first_ord = static_cast<decltype(first_ord)>(tensor->scalar<int64>()());

      TF_RETURN_IF_ERROR(ctx->input("num_records", &tensor));
      auto num_recs = tensor->scalar<int32>()();
      last_ord = first_ord + static_cast<decltype(last_ord)>(num_recs);

      for (auto &header : headers_) {
        header.first_ordinal = first_ord;
        header.last_ordinal = last_ord;
      }

      Tensor *first_ord_out;
      TF_RETURN_IF_ERROR(ctx->allocate_output("first_ordinal_out", TensorShape({}), &first_ord_out));
      first_ord_out->scalar<int64>()() = first_ord;

      return Status::OK();
    }

    chrono::high_resolution_clock::time_point start;
    bool compress_;
    vector<string> record_suffixes_;
    vector<format::FileHeader> headers_;
    string cluster_name;
    string user_name;
    string pool_name;
    string ceph_conf;
    librados::Rados cluster;
    librados::IoCtx io_ctx;
    librados::bufferlist write_buf_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDCephWriteColumnsOp);
} //  namespace tensorflow {
