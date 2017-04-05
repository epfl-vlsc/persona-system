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

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("AGDWriteColumns");
  }


  class AGDWriteColumnsOp : public OpKernel {
  public:
    AGDWriteColumnsOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      using namespace format;
      vector<string> rec_types;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_type", &rec_types));
      for (size_t i = 0; i < rec_types.size(); ++i) {
        auto& t_in = rec_types[i];
        format::FileHeader header;

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
        header.compression_type = CompressionType::UNCOMPRESSED;
        headers_.push_back(header);
      }
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor *path_t, *column_t, *record_id_t;
      OP_REQUIRES_OK(ctx, ctx->input("file_path", &path_t));
      OP_REQUIRES_OK(ctx, ctx->input("column_handle", &column_t));
      OP_REQUIRES_OK(ctx, ctx->input("record_id", &record_id_t));
      auto filepath = path_t->scalar<string>()();
      auto column_vec = column_t->vec<string>();
      auto record_id = record_id_t->scalar<string>()();

      OP_REQUIRES_OK(ctx, InitHeaders(ctx, record_id));

      ResourceContainer<BufferList> *columns;
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(column_vec(0), column_vec(1), &columns));
      core::ScopedUnref column_releaser(columns);
      ResourceReleaser<BufferList> a(*columns);

      auto *buf_list = columns->get();
      auto num_buffers = buf_list->size();

      auto s = Status::OK();

      for (int i = 0; i < num_buffers && s.ok(); ++i) {

        string full_path(filepath + record_suffixes_[i]);

        FILE *file_out = fopen(full_path.c_str(), "w+");
        // TODO get errno out of file
        OP_REQUIRES(ctx, file_out != nullptr, Internal("Unable to open file at path:", full_path));

        OP_REQUIRES_OK(ctx, WriteHeader(ctx, file_out, i));

        size_t fwrite_ret;

        auto &index = (*buf_list)[i].index();
        auto s1 = index.size();
        if (s1 == 0) {
          s = Internal("Parallel column writer got an empty index entry (size 0)!");
        } else {
          fwrite_ret = fwrite(&index[0], s1, 1, file_out);
          if (fwrite_ret != 1) {
            s = Internal("fwrite (uncompressed) gave non-1 return value: ", fwrite_ret);
          } else if (s.ok()) {
            auto &data = (*buf_list)[i].data();

            fwrite_ret = fwrite(&data[0], data.size(), 1, file_out);
            if (fwrite_ret != 1) {
              s = Internal("fwrite (uncompressed) gave non-1 return value: ", fwrite_ret, " when trying to write item of size ", data.size());
            }
          }
        }

        fclose(file_out);
      }

      OP_REQUIRES_OK(ctx, s); // in case s screws up

      Tensor *key_out;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("key_out", TensorShape({}), &key_out));
      key_out->scalar<string>()() = filepath;
    }

  private:

    inline
    Status InitHeaders(OpKernelContext *ctx, const string &record_id) {
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

      if (record_id_ != record_id) {
        auto copy_size = min(sizeof(record_id.size()), sizeof(format::FileHeader::string_id));
        for (auto &header : headers_) {
          strncpy(&header.string_id[0], record_id.c_str(), copy_size);
        }
        record_id_ = record_id;
      }

      return Status::OK();
    }

    inline
    Status WriteHeader(OpKernelContext *ctx, FILE *file_out, int index) {
      auto& header = headers_[index];

      int fwrite_ret = fwrite(&header, sizeof(format::FileHeader), 1, file_out);
      if (fwrite_ret != 1) {
        // TODO get errno out of the file
        fclose(file_out);
        return Internal("fwrite(header_) failed");
      }
      return Status::OK();
    }

    chrono::high_resolution_clock::time_point start;
    vector<string> record_suffixes_;
    string record_id_;
    vector<format::FileHeader> headers_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDWriteColumnsOp);
} //  namespace tensorflow {
