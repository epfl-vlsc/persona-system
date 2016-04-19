#include <vector>
#include <cstdint>
#include "tensorflow/core/framework/writer_op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/writer_base.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/user_ops/dna-align/snap_proto.pb.h"

namespace tensorflow {

REGISTER_OP("DenseWriter")
  .Output("writer_handle: Ref(string)")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("record_name: string")
  .Attr("record_chunk_size: int > 10000")
  .Attr("metadata_path: string")
  .Attr("record_out_dir: string")
  .SetIsStateful()
  .Doc(R"doc(
A writer to convert the input SNAP proto FASTQ file.

Takes a record from the FASTQ reader, splits it into separate, inner redaers, and writes it into files at its chunk intervals.
  )doc");

using namespace std;

namespace {

  class DenseSegmentFileBuffer {
  public:
    void addRecord(const string &record) {
      record_buffer_.insert(record_buffer_.end(), record.begin(), record.end());
      PushBackRelativeIndex(record);
    }

    void clearBuffer() {
      // They should keep their size
      record_buffer_.clear();
      relative_index_.clear();
    }

  protected:
    void PushBackRelativeIndex(const string &record) {
      relative_index_.push_back(static_cast<uint8_t>(record.length()));
    }

    vector<char> record_buffer_;
    vector<uint8_t> relative_index_;
  };

  class BaseSegmentFileBuffer : public DenseSegmentFileBuffer {
  public:
    void addRecord(const string &record) {
      // TODO actual conversion here!
      PushBackRelativeIndex(record);
    }
  };
}

class DenseWriter : public WriterBase {
public:

  DenseWriter(const string& node_name, Env* env, const string& record_name,
              const size_t records_per_chunk, const string& metadata_out_path, const string& record_out_dir)
    : WriterBase(strings::StrCat("DenseWriter'", node_name, "'"), record_name),
      env_(env), records_per_chunk_(records_per_chunk),
      metadata_out_path_(metadata_out_path), record_name_(record_name),
      record_out_dir_(record_out_dir), num_records_(0) {}

  Status WriteLocked(const string& value)
  {
    SnapProto::AlignmentDef alignment;
    SnapProto::ReadDef read_proto;

    if (!alignment.ParseFromString(value)) {
      LOG(ERROR) << "DenseWriter: failed to parse read from protobuf";
      return errors::InvalidArgument("Unable to parse SnapProto alignment def from string: ", value);
    }
    read_proto = alignment.read();
    metadata_buffer_.addRecord(read_proto.meta());
    qualities_buffer_.addRecord(read_proto.qualities());
    bases_buffer_.addRecord(read_proto.bases());

    if (++num_records_ == records_per_chunk_) {
      // TODO actually write out the records
    }
  }

  Status OnWorkStartedLocked(OpKernelContext* context)
  {

  }

  Status OnWorkFinishedLocked()
  {
    
  }

private:

  Env* const env_;
  size_t records_per_chunk_;
  size_t num_records_;
  string metadata_out_path_;
  string record_name_;
  string record_out_dir_;

  // TODO need to add fields to keep state about the records written out already
  // For the json

  DenseSegmentFileBuffer qualities_buffer_;
  DenseSegmentFileBuffer metadata_buffer_;
  BaseSegmentFileBuffer bases_buffer_;
};

class DenseWriterOp : public WriterOpKernel {
public:
  explicit DenseWriterOp(OpKernelConstruction* context)
    : WriterOpKernel(context) {

    string metadata_path, record_out_dir, record_name;
    int records_per_chunk;
    // TODO verify that the paths actually exist using some filesystem tests
    OP_REQUIRES_OK(context,
                   context->GetAttr("metadata_path", &metadata_path));
    OP_REQUIRES_OK(context,
                   context->GetAttr("record_out_dir", &record_out_dir));
    OP_REQUIRES_OK(context,
                   context->GetAttr("record_name", &record_name));
    OP_REQUIRES_OK(context,
                   context->GetAttr("records_per_chunk", &records_per_chunk));
    OP_REQUIRES(context, records_per_chunk > 0,
                errors::InvalidArgument("Records/chunk must be > 0: ", records_per_chunk));

    Env* env = context->env();
    SetWriterFactory([this, metadata_path, env, records_per_chunk,
                      record_out_dir, record_name]() {
                       return new DenseWriter(name(), env, record_name, records_per_chunk, metadata_path, record_out_dir);
      });
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseWriter").Device(DEVICE_CPU),
                        DenseWriterOp);

}  // namespace tensorflow
