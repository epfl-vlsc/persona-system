#include "tensorflow/core/framework/writer_op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/writer_base.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"

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

class DenseWriter : public WriterBase {
public:

  DenseWriter(const string& node_name, Env* env, const string& record_name,
              const size_t records_per_chunk, const string& metadata_out_path, const string& record_out_dir)
    : WriterBase(strings::StrCat("DenseWriter'", node_name, "'"), record_name),
      env_(env), records_per_chunk_(records_per_chunk),
      metadata_out_path_(metadata_out_path), record_name_(record_name),
      record_out_dir_(record_out_dir) {}

  Status WriteLocked(const string& value)
  {
    
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
  string metadata_out_path_;
  string record_name_;
  string record_out_dir_;
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
