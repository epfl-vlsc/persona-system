#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "json/src/json.hpp"
#include <string>
#include <sstream>
#include <fstream>

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using json = nlohmann::json;

  namespace {
    const string op_name("DenseMetadata"), base_ext("base"), qual_ext("qual"), meta_ext("meta"), records_key("records");
    const TensorShape scalar_shape;
  }

  REGISTER_OP(op_name.c_str())
  .SetIsStateful()
  .Attr("output_dir: string")
  .Attr("record_id: string")
  .Input("num_records: int32")
  .Output("first_ordinal: int64")
  .Output("base_path: string")
  .Output("meta_path: string")
  .Output("qual_path: string")
  .Doc(R"doc(
Provides metadata (paths, ordinal assignments) to the column writers
when they receive the column files to write out.

Filesystem directory existence is presumed to be correct and verified
in the calling Python functionality, to only do that querying once.

The primary reason to use this op is to automatically create the json file.
The column writers could create their own filenames given an output_directory path, but this op creates the metadata.json file that makes reading things in easier.
)doc");

  class DenseMetadataOp : public OpKernel {
  public:
    DenseMetadataOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dir", &output_dir_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("record_id", &record_id_));
      path_prefix_ = output_dir_ + "/" + record_id_ + "_";
      metadata_["name"] = record_id_;
      metadata_["version"] = 2;
      metadata_[records_key] = json::array();
    }

    ~DenseMetadataOp() {
      WriteMetadata();
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->scalar<int32>()();

      Tensor *first_ordinal, *base_path, *meta_path, *qual_path;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("first_ordinal", scalar_shape, &first_ordinal));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("base_path", scalar_shape, &base_path));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("meta_path", scalar_shape, &meta_path));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("qual_path", scalar_shape, &qual_path));

      first_ordinal->scalar<int64>()() = ordinal_;

      json path_tuple, chunk_record;
      chunk_record["first"] = ordinal_;
      chunk_record["last"] = ordinal_ + num_records; // TODO should this maybe be -1?

      stringstream prefix;
      prefix << path_prefix_ << ordinal_ << ".";
      string full_prefix(prefix.str());

      WritePathRecord(path_tuple, base_ext, full_prefix, base_path);
      WritePathRecord(path_tuple, qual_ext, full_prefix, qual_path);
      WritePathRecord(path_tuple, meta_ext, full_prefix, meta_path);

      chunk_record["path"] = path_tuple;

      metadata_[records_key].push_back(chunk_record);

      ordinal_ += num_records;
    }
  private:

    void WriteMetadata()
    {
      output_dir_.append("/metadata.json"); // just reuse the member var
      ofstream metadata_file(output_dir_, ofstream::out | ofstream::trunc);
      if (!metadata_file) {
        LOG(ERROR) << "Unable to write out metadata file at path: " << output_dir_;
        return;
      }

      metadata_file << metadata_;
      metadata_file.close();
    }

    inline void WritePathRecord(json &path, const string &ext_key, const string &full_prefix, Tensor *tensor) {
      auto pth = full_prefix + ext_key;
      path[ext_key] = pth;
      tensor->scalar<string>()() = pth;
    }

    uint64_t ordinal_ = 0;
    string output_dir_, record_id_, path_prefix_;
    json metadata_;
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), DenseMetadataOp);
} // namespace tensorflow {
