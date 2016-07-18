#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/dense-format/buffer.h"
#include "tensorflow/core/user_ops/dense-format/column_builder.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/FileFormat.h"
#include "tensorflow/core/user_ops/dense-format/column_builder.h"
#include "GenomeIndex.h"
#include "Read.h"
#include "snap_proto.pb.h"
#include "SnapAlignerWrapper.h"
#include "genome_index_resource.h"
#include "snap_read_decode.h"
#include "snap_results_decode.h"
#include "aligner_options_resource.h"
#include "tensorflow/core/user_ops/dense-format/read_resource.h"

namespace tensorflow {
using namespace std;
using namespace errors;

class SnapAlignDenseOp : public OpKernel {
  public:
    explicit SnapAlignDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    ~SnapAlignDenseOp() override {
      core::ScopedUnref index_unref(index_resource_);
      core::ScopedUnref options_unref(options_resource_);
      core::ScopedUnref buf_pool_unref(buf_pool_);
    }

  Status InitHandles(OpKernelContext* ctx)
  {
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "options_handle", &options_resource_));
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "genome_handle", &index_resource_));
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pool", &buf_pool_));
    TF_RETURN_IF_ERROR(snap_wrapper::init());

    LOG(INFO) << "SNAP Kernel creating BaseAligner";

    base_aligner_ = snap_wrapper::createAligner(index_resource_->get_index(), options_resource_->value());

    AlignerOptions* options = options_resource_->value();

    if (options->maxSecondaryAlignmentAdditionalEditDistance < 0) {
      num_secondary_alignments_ = 0;
    } else {
      num_secondary_alignments_ = BaseAligner::getMaxSecondaryResults(options->numSeedsFromCommandLine,
                                                                      options->seedCoverage, MAX_READ_LENGTH, options->maxHits, index_resource_->get_index()->getSeedLength());
    }

    return Status::OK();
  }

  Status GetResultBuffer(OpKernelContext* ctx, ResourceContainer<Buffer> **ctr)
  {
    TF_RETURN_IF_ERROR(buf_pool_->GetResource(ctr));
    (*ctr)->get()->reset();
    TF_RETURN_IF_ERROR((*ctr)->allocate_output("result_buf_handle", ctx));
    return Status::OK();
  }

  size_t InitializeReads(const SnapReadDecode &read_batch)
  {
    size_t num_reads = read_batch.size();

    input_reads_.clear(); input_reads_.reserve(num_reads);

    for (size_t i = 0; i < num_reads; i++) {
      if (read_batch.bases_len(i) == 0) {
        LOG(INFO) << "string was empty, is this a partial batch?";
        continue;
      }
      /*LOG(INFO) << "Read from decoder:";
        LOG(INFO) << "Meta: " << read_batch.metadata(i) <<  "\nBases: " << read_batch.bases(i)
        << "\nQual: " << read_batch.qualities(i) << "\nbase len: " << read_batch.bases_len(i);*/

      Read snap_read;
      snap_read.init(
                      read_batch.metadata(i),  // id
                      read_batch.metadata_len(i), // id len
                      read_batch.bases(i),  // data (bases)
                      read_batch.qualities(i),  // qualities
                      read_batch.bases_len(i)  // data len
                      );

      input_reads_.push_back(snap_read);
    }

    return input_reads_.size();
  }

  size_t InitializeReadsDense(ReadResource *reads)
  {
    reads->reset_iter();
    Status status;

    while (status.ok())
    {
      const char *bases, *qualities, *metadata;
      std::size_t bases_len, qualities_len, metadata_len; 
      status = reads->get_next_record(&bases, &bases_len, &qualities, 
          &qualities_len, &metadata, &metadata_len);

      if (status.ok())
      {
        Read snap_read;
        snap_read.init(metadata, metadata_len, bases, qualities, bases_len);
        input_reads_.push_back(snap_read);
      }
    }

    return input_reads_.size(); 
  }

    void Compute(OpKernelContext* ctx) override {
      if (base_aligner_ == nullptr) {
        OP_REQUIRES_OK(ctx, InitHandles(ctx));
      }

      ResourceContainer<Buffer> *buffer_resource_container;
      OP_REQUIRES_OK(ctx, GetResultBuffer(ctx, &buffer_resource_container));
      auto &alignment_result_buffer = buffer_resource_container->get()->get();

      ResourceContainer<ReadResource> *reads;
      const Tensor *read_input;
      OP_REQUIRES_OK(ctx, ctx->input("read", &read_input)); 
      auto data = read_input->vec<string>(); // data(0) = container, data(1) = name 
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &reads)); 

      size_t num_actual_reads =  InitializeReadsDense(reads->get()); 

      auto options = options_resource_->value();

      vector<SingleAlignmentResult> alignment_results;

      bool first_is_primary;
      cigarString_.clear();

      for (size_t i = 0; i < num_actual_reads; ++i) {
        OP_REQUIRES_OK(ctx, snap_wrapper::alignSingle(base_aligner_, options, &input_reads_[i],
                                                      &alignment_results, num_secondary_alignments_, first_is_primary));

        size_t num_results = alignment_results.size();
        OP_REQUIRES(ctx, num_results == 1, Internal("New format only supports exactly 1 result. Time to fix it :)"));

        const Genome *genome = index_resource_->get_genome();
        SAMFormat format(options->useM);
        flag_ = 0;

        // compute the CIGAR strings and flags
        // input_reads[i] holds the current snap_read
        OP_REQUIRES_OK(ctx, snap_wrapper::computeCigarFlags(
          &input_reads_[i], alignment_results, alignment_results.size(), first_is_primary, format,
          options->useM, lvc_, genome, cigarString_, flag_));

        result_builder_.AppendAlignmentResult(alignment_results[0], cigarString_, alignment_result_buffer);

        alignment_results.clear();

    }

#ifdef NEW_OUTPUT
      result_builder_.AppendAndFlush(alignment_result_buffer);
#endif
    }

  private:
    BaseAligner* base_aligner_ = nullptr;
    ReferencePool<Buffer> *buf_pool_ = nullptr;
    int num_secondary_alignments_ = 0;
    GenomeIndexResource* index_resource_ = nullptr;
    AlignerOptionsResource* options_resource_ = nullptr;
    const FileFormat *format_;

    string cigarString_;
    AlignmentResultBuilder result_builder_;
    int flag_;
    LandauVishkinWithCigar lvc_;

    vector<Read> input_reads_; // a vector to pass to SNAP
};


  REGISTER_OP("SnapAlignDense")
      .Input("genome_handle: Ref(string)")
      .Input("options_handle: Ref(string)")
      .Input("buffer_pool: Ref(string)")
      .Input("read: string")
      .Output("result_buf_handle: string")
      .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
output: a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.
)doc");


    REGISTER_KERNEL_BUILDER(Name("SnapAlignDense").Device(DEVICE_CPU), SnapAlignDenseOp);

}  // namespace tensorflow
