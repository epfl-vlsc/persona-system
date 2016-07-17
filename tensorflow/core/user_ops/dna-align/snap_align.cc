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
#include "GenomeIndex.h"
#include "Read.h"
#include "snap_proto.pb.h"
#include "SnapAlignerWrapper.h"
#include "genome_index_resource.h"
#include "snap_read_decode.h"
#include "snap_results_decode.h"
#include "aligner_options_resource.h"
#include <boost/timer/timer.hpp>

namespace tensorflow {
using namespace std;

class SnapAlignOp : public OpKernel {
  public:
    explicit SnapAlignOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    ~SnapAlignOp() override {
      if (index_resource_) index_resource_->Unref();
      if (options_resource_) options_resource_->Unref();
    }

    void Compute(OpKernelContext* ctx) override {
      //LOG(INFO) << "SnapAlign started";

      if (base_aligner_ == nullptr) {
        OP_REQUIRES_OK(ctx,
            GetResourceFromContext(ctx, "options_handle", &options_resource_));
        OP_REQUIRES_OK(ctx,
            GetResourceFromContext(ctx, "genome_handle", &index_resource_));

        OP_REQUIRES_OK(ctx,
            snap_wrapper::init());
        LOG(INFO) << "SNAP Kernel creating BaseAligner";

        base_aligner_ = snap_wrapper::createAligner(index_resource_->get_index(), options_resource_->value());

        AlignerOptions* options = options_resource_->value();

        if (options->maxSecondaryAlignmentAdditionalEditDistance < 0) {
          num_secondary_alignments_ = 0;
        }
        else {
          num_secondary_alignments_ = BaseAligner::getMaxSecondaryResults(options->numSeedsFromCommandLine,
              options->seedCoverage, MAX_READ_LENGTH, options->maxHits, index_resource_->get_index()->getSeedLength());
        }
      }

      //boost::timer::auto_cpu_timer t;
      auto begin = std::chrono::high_resolution_clock::now();

      const Tensor* reads;
      OP_REQUIRES_OK(ctx, ctx->input("read", &reads));
      
      //LOG(INFO) << "reads shape is: " << reads->shape().DebugString();
      SnapReadDecode read_batch(reads);
      size_t num_reads = read_batch.size();
      size_t reads_encountered = num_reads;
      
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, 
            SnapResultsDecode::get_results_shape(num_reads, 
              num_secondary_alignments_), &out));

      MutableSnapResultsDecode results(out);
      
      bool first_is_primary;
      AlignerOptions* options = options_resource_->value();

      for (size_t i = 0; i < num_reads; i++) {

        if (read_batch.bases_len(i) == 0) {
          reads_encountered = i;
          LOG(INFO) << "string was empty, is this a partial batch?";
          break;
        }
        /*LOG(INFO) << "Read from decoder:";
        LOG(INFO) << "Meta: " << read_batch.metadata(i) <<  "\nBases: " << read_batch.bases(i)
          << "\nQual: " << read_batch.qualities(i) << "\nbase len: " << read_batch.bases_len(i);*/
      
        snap_read_.init(
            read_batch.metadata(i),  // id
            read_batch.metadata_len(i), // id len
            read_batch.bases(i),  // data (bases)
            read_batch.qualities(i),  // qualities
            read_batch.bases_len(i)  // data len
            );
        
        snap_read_.clip(options->clipping);

        if (!passesReadFilter(&snap_read_, options_resource_->value())) {
          results.set_first_is_primary(i, true);
          results.set_num_results(i, 1); 
          results.set_result_type(i, 1, (int64)AlignmentResult::NotFound); // cast from enum
          results.set_genome_location(i, 1, InvalidGenomeLocation);
          results.set_score(i, 1, 0);
          results.set_mapq(i, 1, 0);
          results.set_direction(i, 1, 0);
          continue;
        }

        SingleAlignmentResult primaryResult;
        int num_secondary_alignments = 0;
        int num_secondary_results;
        base_aligner_->AlignRead(
          &snap_read_,
          &primaryResult,
          options->maxSecondaryAlignmentAdditionalEditDistance,
          num_secondary_alignments * sizeof(SingleAlignmentResult),
          &num_secondary_results,
          num_secondary_alignments,
          nullptr //secondaryResults
        );

        if (options->passFilter(&snap_read_, primaryResult.status, false, false)) {
          first_is_primary = true;
        }
        else {
          first_is_primary = false;
        }

        results.set_first_is_primary(i, first_is_primary);
        results.set_num_results(i, 1); 
        results.set_result_type(i, 0, (int64)primaryResult.status); // cast from enum
        results.set_genome_location(i, 0, GenomeLocationAsInt64(primaryResult.location));
        results.set_score(i, 0, primaryResult.score);
        results.set_mapq(i, 0, primaryResult.mapq);
        results.set_direction(i, 0, primaryResult.direction);

      }


      for (size_t i = reads_encountered; i < num_reads; i++) {
        // for uneven batches, set the num of results to 0
        LOG(INFO) << "setting 0 for uneven batch";
        results.set_num_results(i, 0);
      }
      
      OP_REQUIRES_OK(ctx, ctx->set_output("reads_out", *reads));
      
      auto end = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "snap align time is: " << ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count())/1000000000.0f;

      //LOG(INFO) << "actual: " << num_actual_reads << " total: " << num_reads;
    }

  private:

    bool passesReadFilter(Read* read, AlignerOptions* options) {
      return read->getDataLength() >= options->minReadLength && read->countOfNs() <= options->maxDist;
    }

    BaseAligner* base_aligner_ = nullptr;
    int num_secondary_alignments_ = 0;
    GenomeIndexResource* index_resource_ = nullptr;
    AlignerOptionsResource* options_resource_ = nullptr;
    Read snap_read_;

};


  REGISTER_OP("SnapAlign")
      .Input("genome_handle: Ref(string)")
      .Input("options_handle: Ref(string)")
      .Input("read: string")
      .Output("output: int64")
      .Output("reads_out: string")
      .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
output: a tensor [num_reads] containing serialized reads and results
containing the alignment candidates. 
)doc");


    REGISTER_KERNEL_BUILDER(Name("SnapAlign").Device(DEVICE_CPU), SnapAlignOp);

}  // namespace tensorflow
