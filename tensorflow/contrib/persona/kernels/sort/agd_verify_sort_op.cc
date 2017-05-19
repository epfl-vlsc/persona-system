#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  // TODO should probalby put this in an H file somewhere
  inline bool operator<(const Position& lhs, const Position& rhs) {
    if (lhs.ref_index() < rhs.ref_index()) {
      return true;
    } else if (lhs.ref_index() == rhs.ref_index()) {
      if (lhs.position() < rhs.position()) return true;
      else return false;
    } else
      return false;
  }


  class AGDVerifySortOp : public OpKernel {
  public:
    AGDVerifySortOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    ~AGDVerifySortOp() {
    }

    Status LoadChunk(OpKernelContext* ctx, string chunk_path) {

      LOG(INFO) << "chunk path is " << chunk_path;
      TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile(path_ + chunk_path + ".results", &mmap_results_));
      results_buf_.reset();
      TF_RETURN_IF_ERROR(rec_parser_.ParseNew((const char*)mmap_results_->data(), mmap_results_->length(),
            false, &results_buf_, &results_ord_, &num_results_, record_id_));
      return Status::OK();
    }

    void Compute(OpKernelContext* ctx) override {

      const Tensor *chunk_names_t, *path_t, *chunk_size_t;
      OP_REQUIRES_OK(ctx, ctx->input("chunk_names", &chunk_names_t));
      OP_REQUIRES_OK(ctx, ctx->input("path", &path_t));
      OP_REQUIRES_OK(ctx, ctx->input("chunk_size", &chunk_size_t));
      auto chunk_names = chunk_names_t->vec<string>();
      path_ = path_t->scalar<string>()();
      auto chunksize = chunk_size_t->scalar<int>()();

      Status status;
      Position prev_position;
      prev_position.set_ref_index(-1);
      prev_position.set_position(-1);
      int index = 0;
      Alignment agd_result;

      for (int i = 0; i < chunk_names.size(); i++) {

        OP_REQUIRES_OK(ctx, LoadChunk(ctx, chunk_names(i)));
        AGDResultReader results_reader(results_buf_.data(), (size_t)chunksize);
        index = 0;


        status = results_reader.GetNextResult(agd_result);
        while (status.ok()) {
          //agd_result = reinterpret_cast<const format::AlignmentResult*>(data);
          //LOG(INFO) << "AGD location is: " << agd_result->location_;
          if (agd_result.position() < prev_position) {
            LOG(INFO) << "AGD SET IS NOT SORTED. Offending entry in chunk " << i 
              << " at index " << index << ". Prev: " << prev_position.DebugString() << " Curr: "
              << agd_result.position().DebugString();

            return;
          }
          index++;
          prev_position = agd_result.position();
          status = results_reader.GetNextResult(agd_result);
        }
      }

      LOG(INFO) << "AGD Dataset is sorted.";
      mmap_results_.reset(nullptr);

    }

  private:
    std::unique_ptr<ReadOnlyMemoryRegion> mmap_results_;
    Buffer results_buf_;
    uint64_t results_ord_;
    uint32_t num_results_;
    RecordParser rec_parser_;
    string path_, record_id_;
  };

  REGISTER_KERNEL_BUILDER(Name("AGDVerifySort").Device(DEVICE_CPU), AGDVerifySortOp);
} //  namespace tensorflow {
