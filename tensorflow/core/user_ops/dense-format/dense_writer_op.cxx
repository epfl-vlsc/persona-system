#include <vector>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include "tensorflow/core/framework/writer_op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/writer_base.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/object_pool.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/user_ops/dna-align/snap_proto.pb.h"
#include "format.h"

namespace tensorflow {

REGISTER_OP("DenseWriter")
  .Output("writer_handle: Ref(string)")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("record_name: string")
  .Attr("record_chunk_size: int > 10000")
  .Attr("metadata_path: string")
  .Attr("record_out_dir: string")
  .Attr("parallelism: int = 2")
  .SetIsStateful()
  .Doc(R"doc(
A writer to convert the input SNAP proto FASTQ file.

Takes a record from the FASTQ reader, splits it into separate, inner redaers, and writes it into files at its chunk intervals.
  )doc");

using namespace std;

namespace {
  typedef shared_ptr<vector<uint8_t>> IndexBufferPtr;

  class IndexBuffer {
  public:
    void AddRecord(const size_t length) {
      relative_index_->push_back(static_cast<uint8_t>(length));
    }

    void setNewBuffer(IndexBufferPtr relative_index) {
      relative_index_ = relative_index;
    }
  private:
    IndexBufferPtr relative_index_;
  };

  class DenseSegmentFileBuffer {
  public:
    typedef shared_ptr<vector<char>> DataBufferPtr;

    Status addRecord(const string &record) {
      record_buffer_->insert(record_buffer_->end(), record.begin(), record.end());
      relative_index_.AddRecord(record.length());
      return Status::OK();
    }

    void setNewBuffers(DataBufferPtr record_buffer, IndexBufferPtr index_buffer) {
      // They should keep their size of the allocated buffer underneath
      record_buffer_ = record_buffer;
      relative_index_.setNewBuffer(index_buffer);
    }

  private:
    DataBufferPtr record_buffer_;
    IndexBuffer relative_index_;
  };

  class BaseSegmentFileBuffer {
  public:
    typedef shared_ptr<vector<format::BinaryBases>> DataBufferPtr;

    Status addRecord(const string &record) {
      const size_t old_buffer_size = base_buffer_->size();
      TF_RETURN_IF_ERROR(format::BinaryBaseRecord::IntoBases(record.c_str(),
                                                             record.length(),
                                                             *base_buffer_));
      const size_t new_bases = base_buffer_->size() - old_buffer_size;

      if (new_bases < 1) {
        return errors::InvalidArgument("Appending base record '", record, "' to BaseSegmentFileBuffer resulted in no new bases");
      }

      relative_index_.AddRecord(new_bases * sizeof(format::BinaryBases));
    }

    void setNewBuffers(DataBufferPtr base_buffer, IndexBufferPtr index_buffer) {
      // They should keep their size of the allocated buffer underneath
      base_buffer_ = base_buffer;
      relative_index_.setNewBuffer(index_buffer);
    }

  private:
    DataBufferPtr base_buffer_;
    IndexBuffer relative_index_;
  };

  template <typename T>
  struct SegmentLoan {
    typedef shared_ptr<vector<T>> DataBufferPtr;
    typedef pair<DataBufferPtr, function<void()>> DataLoan;
    typedef pair<IndexBufferPtr, function<void()>> IndexLoan;

    SegmentLoan(
                DataLoan data_loan_,
                IndexLoan index_loan_
                ) : data_loan(data_loan_.first),
      data_loan_releaser(data_loan_.second),
      index_loan(index_loan_.first),
      index_loan_releaser(index_loan_.second) {}

    SegmentLoan() : data_loan(nullptr), index_loan(nullptr),
                    data_loan_releaser([](){}),
                    index_loan_releaser([](){}) {}

    void setDataLoan(DataLoan data_loan_) {
      data_loan = data_loan_.first;
      data_loan_releaser = data_loan_.second;
    }

    void setIndexLoan(IndexLoan index_loan_) {
      index_loan = index_loan_.first;
      index_loan_releaser = index_loan_.second;
    }

    DataBufferPtr data_loan;
    function<void()> data_loan_releaser;
    IndexBufferPtr index_loan;
    function<void()> index_loan_releaser;
  };
} // namespace

class DenseWriter : public WriterBase {
public:

  DenseWriter(const string& node_name, Env* env, const string& record_name,
              const size_t records_per_chunk, const string& metadata_out_path, const string& record_out_dir,
              const size_t parallelism)
    : WriterBase(strings::StrCat("DenseWriter'", node_name, "'"), record_name),
      env_(env), records_per_chunk_(records_per_chunk),
      metadata_out_path_(metadata_out_path), record_name_(record_name),
      record_out_dir_(record_out_dir), num_records_(0), parallelism_(parallelism),
      char_buffer_pool_(parallelism * 2, []() { return new vector<char>(); }),
      bases_buffer_pool_(parallelism * 3, []() { return new vector<format::BinaryBases>(); }),
      rel_index_buffer_pool_(parallelism, []() { return new vector<uint8_t>(); }),
      thread_pool_(nullptr) {}

  Status WriteLocked(const string& value)
  {
    SnapProto::AlignmentDef alignment;
    SnapProto::ReadDef read_proto;

    if (!alignment.ParseFromString(value)) {
      LOG(ERROR) << "DenseWriter: failed to parse read from protobuf";
      return errors::InvalidArgument("Unable to parse SnapProto alignment def from string: ", value);
    }
    read_proto = alignment.read();

    // If we've started a new chunk, write out old chunks and start
    if (SetCurrentChunkName(read_proto.record_name())) {
      WriteChunkFiles();
    }

    metadata_buffer_.addRecord(read_proto.meta());
    qualities_buffer_.addRecord(read_proto.qualities());
    bases_buffer_.addRecord(read_proto.bases());

    if (++num_records_ == records_per_chunk_) {
      WriteChunkFiles();
    }
  }

  Status OnWorkStartedLocked(OpKernelContext* context)
  {
    if (!thread_pool_) {
      // Set up a threadpool. Only need to do it the first time
      thread_pool_ = move(unique_ptr<thread::ThreadPool>(
                      new thread::ThreadPool(context->env(),
                                             strings::StrCat("dense_writer_", SanitizeThreadSuffix(context->op_kernel().name())),
                                             parallelism_ * 3)
                                                         )); // 3 for bases, qualities, and metadata writing
    }
  }

  Status OnWorkFinishedLocked()
  {
    if (num_records_ > 0) {
      WriteChunkFiles();
    }
  }

private:

  bool SetCurrentChunkName(const string& new_name)
  {
    if (current_chunk_name_.length() == 0) {
      current_chunk_name_ = new_name;
    } else if (current_chunk_name_.compare(new_name)) {
      current_chunk_name_ = new_name;
      return true;
    }
    return false;
  }

  Status WriteChunkFiles()
  {
    // TODO need to fire this off!
    num_records_ = 0;
  }

  Env* const env_;
  size_t records_per_chunk_;
  size_t num_records_;
  size_t parallelism_;
  string metadata_out_path_;
  string record_name_;
  string record_out_dir_;
  string current_chunk_name_;

  // TODO need to add fields to keep state about the records written out already
  // For the json

  DenseSegmentFileBuffer qualities_buffer_;
  SegmentLoan<char> qualities_loan;
  DenseSegmentFileBuffer metadata_buffer_;
  SegmentLoan<char> metadata_loan;
  BaseSegmentFileBuffer bases_buffer_;
  SegmentLoan<format::BinaryBases> bases_loan;

  ObjectPool<vector<char>> char_buffer_pool_;
  ObjectPool<vector<uint8_t>> rel_index_buffer_pool_;
  ObjectPool<vector<format::BinaryBases>> bases_buffer_pool_;

  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class DenseWriterOp : public WriterOpKernel {
public:
  explicit DenseWriterOp(OpKernelConstruction* context)
    : WriterOpKernel(context) {

    string metadata_path, record_out_dir, record_name;
    int records_per_chunk, parallelism;
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
    OP_REQUIRES_OK(context,
                   context->GetAttr("parallelism", &parallelism));
    OP_REQUIRES(context, parallelism > 0,
                errors::InvalidArgument("Parallelism for DenseReader must be > 0: ",
                                        parallelism));

    Env* env = context->env();
    SetWriterFactory([this, metadata_path, env, records_per_chunk,
                      record_out_dir, record_name, parallelism]() {
                       return new DenseWriter(name(), env, record_name, records_per_chunk, metadata_path, record_out_dir, parallelism);
      });
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseWriter").Device(DEVICE_CPU), DenseWriterOp);

}  // namespace tensorflow
