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
#include "SnapAlignerWrapper.h"
#include "genome_index_resource.h"
#include "aligner_options_resource.h"
#include "tensorflow/core/user_ops/dense-format/read_resource.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"

namespace tensorflow {
using namespace std;
using namespace errors;

class SnapAlignDenseOp : public OpKernel {
  public:
    explicit SnapAlignDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
    }

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

      options_ = options_resource_->value();
      genome_ = index_resource_->get_genome();

      if (options_->maxSecondaryAlignmentAdditionalEditDistance < 0) {
        num_secondary_alignments_ = 0;
      } else {
        num_secondary_alignments_ = BaseAligner::getMaxSecondaryResults(options_->numSeedsFromCommandLine,
            options_->seedCoverage, MAX_READ_LENGTH, options_->maxHits, index_resource_->get_index()->getSeedLength());
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

    void Compute(OpKernelContext* ctx) override {
      if (base_aligner_ == nullptr) {
        OP_REQUIRES_OK(ctx, InitHandles(ctx));
      }

      //auto begin = std::chrono::high_resolution_clock::now();

      ResourceContainer<Buffer> *buffer_resource_container;
      OP_REQUIRES_OK(ctx, GetResultBuffer(ctx, &buffer_resource_container));
      auto &alignment_result_buffer = buffer_resource_container->get()->get();

      ResourceContainer<ReadResource> *reads_container;
      const Tensor *read_input;
      OP_REQUIRES_OK(ctx, ctx->input("read", &read_input)); 
      auto data = read_input->vec<string>(); // data(0) = container, data(1) = name 
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &reads_container)); 
      //LOG(INFO) << "aligner doing " << num_actual_reads << " reads";

      // TODO call reads->split(chunk_size, ...) here

      core::ScopedUnref a(reads_container);
      ResourceReleaser<ReadResource> b(*reads_container);
      auto reads = reads_container->get();
      {
        auto start = clock();
        ReadResourceReleaser r(*reads);
        bool first_is_primary = true; // we only ever generate one result
        const char *bases, *qualities;
        std::size_t bases_len, qualities_len;
        SingleAlignmentResult primaryResult;
        int num_secondary_alignments = 0;
        int num_secondary_results;
        SAMFormat format(options_->useM);


        while (reads->get_next_record(&bases, &bases_len, &qualities, &qualities_len).ok()) {
          snap_read_.init(nullptr, 0, bases, qualities, bases_len);
          snap_read_.clip(options_->clipping);
          if (snap_read_.getDataLength() < options_->minReadLength || snap_read_.countOfNs() > options_->maxDist) {
            if (!options_->passFilter(&snap_read_, AlignmentResult::NotFound, true, false)) {
              LOG(INFO) << "FILTERING READ";
            } else {
              primaryResult.status = AlignmentResult::NotFound;
              primaryResult.location = InvalidGenomeLocation;
              primaryResult.mapq = 0;
              primaryResult.direction = FORWARD;
              cigarString_.clear();
              result_builder_.AppendAlignmentResult(primaryResult, "*", 4, alignment_result_buffer);
              continue;
            }
          }


          base_aligner_->AlignRead(
            &snap_read_,
            &primaryResult,
            options_->maxSecondaryAlignmentAdditionalEditDistance,
            num_secondary_alignments * sizeof(SingleAlignmentResult),
            &num_secondary_results,
            num_secondary_alignments,
            nullptr //secondaryResults
          );

          // we may need to do post process options->passfilter here?

          // compute the CIGAR strings and flags and adjust location according to clipping
          // input_reads[i] holds the current snap_read
          flag_ = 0;
          cigarString_.clear();
          
          OP_REQUIRES_OK(ctx, snap_wrapper::adjustResults(
                &snap_read_, primaryResult, first_is_primary, format,
                options_->useM, lvc_, genome_, cigarString_, flag_));

          /*LOG(INFO) << " result: location " << primaryResult.location <<
            " direction: " << primaryResult.direction << " score " << primaryResult.score << " cigar: " 
            << cigarString_ << " mapq: " << primaryResult.mapq;*/

          result_builder_.AppendAlignmentResult(primaryResult, cigarString_, flag_, alignment_result_buffer);
        }
        //LOG(INFO) << "done aligning";

        result_builder_.AppendAndFlush(alignment_result_buffer);

        //LOG(INFO) << "done append";
        //auto end = std::chrono::high_resolution_clock::now();
        //LOG(INFO) << "snap align time is: " << ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count())/1000000000.0f;
        tracepoint(bioflow, snap_align_kernel, clock() - start);
      }
      chunk_handles_.clear(); // release all resources
    }

  private:
    BaseAligner* base_aligner_ = nullptr;
    ReferencePool<Buffer> *buf_pool_ = nullptr;
    int num_secondary_alignments_ = 0, chunk_size_;
    GenomeIndexResource* index_resource_ = nullptr;
    AlignerOptionsResource* options_resource_ = nullptr;
    const FileFormat *format_;
    const Genome *genome_; 
    AlignerOptions* options_;
    Read snap_read_;

    string cigarString_;
    AlignmentResultBuilder result_builder_;
    int flag_;
    LandauVishkinWithCigar lvc_;

    vector<Read> input_reads_; // a vector to pass to SNAP
    vector<unique_ptr<ReadResource>> chunk_handles_;
};


REGISTER_OP("SnapAlignDense")
  .Attr("chunk_size: int = 10000")
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

chunk_size is the number of records to chunk incoming dense reads into
)doc");


  REGISTER_KERNEL_BUILDER(Name("SnapAlignDense").Device(DEVICE_CPU), SnapAlignDenseOp);

}  // namespace tensorflow
