#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "data.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <utility>
#include <sys/types.h>
#include <sys/stat.h>
#include "format.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;

  namespace {
    const string op_name("ColumnWriter");
  }

  REGISTER_OP(op_name.c_str())
  .Attr("record_id: string")
  .Attr("record_types: list({'base', 'qual', 'metadata', 'results'})")
  .Attr("output_dir: string = ''")
  .Input("columns: string")
  .Input("file_path: string")
  .Input("first_ordinal: int64")
  .Input("num_records: int32")
  .Output("file_path_out: string")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
      using namespace shape_inference;

      vector<string> record_types;
      TF_RETURN_IF_ERROR(c->GetAttr("record_types", &record_types));

      ShapeHandle input_data;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_data));
      auto dim_handle = c->Dim(input_data, 1);
      auto dim_value = c->Value(dim_handle);
      if (dim_value != 2) {
        return Internal("columns must be an Nx2 matrix, but got ", dim_value, " for the 2nd dim");
      }

      dim_handle = c->Dim(input_data, 0);
      dim_value = c->Value(dim_handle);
      auto expected_size = record_types.size();
      if (dim_value != expected_size) {
        return Internal("columns must have ", expected_size, " in 0 dim, but got ", dim_value);
      }

      for (int i = 1; i < 4; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &input_data));
      }

      c->set_output(0, c->input(1));

      return Status::OK();
    })
  .Doc(R"doc(
)doc");

  class ColumnWriterOp : public OpKernel {
  public:
    ColumnWriterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      string s;
      FileHeader header;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_id", &s));
      auto max_size = sizeof(header.string_id);
      OP_REQUIRES(ctx, s.length() < max_size,
                  Internal("record_id for column header '", s, "' greater than 32 characters"));

      vector<string> rec_types;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_types", &rec_types));

      RecordType t;
      for (const auto &record_type : rec_types) {
        if (s.compare("base") == 0) {
          t = RecordType::BASES;
        } else if (s.compare("qual") == 0) {
          t = RecordType::QUALITIES;
        } else if (s.compare("metadata") == 0) {
          t = RecordType::COMMENTS;
        } else { // no need to check. we're saved by string enum types if TF
          t = RecordType::ALIGNMENT;
        }
        header.record_type = static_cast<uint8_t>(t);
        strncpy(header.string_id, s.c_str(), max_size);
        header_infos_.push_back(make_pair(header, "." + record_type));
      }

      OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dir", &s));
      if (!s.empty()) {
        if (s.back() != '/') {
          s.push_back('/');
        }
        struct stat outdir_info;
        OP_REQUIRES(ctx, stat(s.c_str(), &outdir_info) == 0, Internal("Unable to stat path: ", s));
        OP_REQUIRES(ctx, S_ISDIR(outdir_info.st_mode), Internal("Path ", s, " is not a directory"));
        record_prefix_path_ = s;
      }
    }

    void Compute(OpKernelContext* ctx) override {
      OP_REQUIRES_OK(ctx, InitHeaders(ctx));

      const Tensor *file_path_t;
      OP_REQUIRES_OK(ctx, ctx->input("file_path", &file_path_t));
      auto &file_path = file_path_t->scalar<string>()();

      string full_path(record_prefix_path_ + file_path);

      const Tensor *columns_t;
      OP_REQUIRES_OK(ctx, ctx->input("columns", &columns_t));
      auto columns = columns_t->matrix<string>();

      ResourceContainer<Data> *column_data;
      Data *data;
      auto *rmgr = ctx->resource_manager();
      for (size_t i = 0; i < header_infos_.size(); i++) {
        auto &header_info = header_infos_[i];
        OP_REQUIRES_OK(ctx, rmgr->Lookup(columns(i, 0), columns(i, 1), &column_data));
        core::ScopedUnref column_releaser(column_data);
        {
          ResourceReleaser<Data> rr(*column_data);
          data = column_data->get();

          OP_REQUIRES_OK(ctx, WriteFile(file_path, data, header_info));

          data->release();
        }
      }

      Tensor *file_path_out;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_path_out", file_path_t->shape(), &file_path_out));
      file_path_out->scalar<string>()() = file_path;
    }

  private:

    bool compress_;
    string record_prefix_path_, path_scratch_;
    vector<tuple<format::FileHeader, string>> header_infos_;

    Status WriteFile(const string &file_path, Data *data, decltype(header_infos_)::value_type &header_info) {
      auto &header = get<0>(header_info);
      auto &file_suffix = get<1>(header_info);

      path_scratch_ = record_prefix_path_ + file_path + file_suffix;

      FILE *file_out = fopen(path_scratch_.c_str(), "w+");
      if (!file_out) {
        return Internal("fopen(", path_scratch_, ") returned null");
      }

      int ret = fwrite(&header, sizeof(header), 1, file_out);
      if (ret != 1) {
        fclose(file_out);
        return Internal("fwrite of header to ", path_scratch_, " returned non-1 code ", ret);
      }

      ret = fwrite(data->data(), data->size(), 1, file_out);
      if (ret != 1) {
        fclose(file_out);
        return Internal("fwrite of data to ", path_scratch_, " returned non-1 code ", ret);
      }

      fclose(file_out);

      return Status::OK();
    }

    Status InitHeaders(OpKernelContext *ctx) {
      const Tensor *first_ord_t, *num_recs_t;
      TF_RETURN_IF_ERROR(ctx->input("first_ordinal", &first_ord_t));
      TF_RETURN_IF_ERROR(ctx->input("num_records", &num_recs_t));
      uint64_t first_ord = static_cast<uint64_t>(first_ord_t->scalar<int64>()());
      uint64_t num_recs = static_cast<uint64_t>(num_recs_t->scalar<int32>()());
      uint64_t last_ord = first_ord + num_recs;

      for (auto &header_info : header_infos_) {
        auto &header = get<0>(header_info);
        header.first_ordinal = first_ord;
        header.last_ordinal = last_ord;
      }

      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), ColumnWriterOp);
} //  namespace tensorflow {
