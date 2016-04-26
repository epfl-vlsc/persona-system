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

      const Tensor* reads;
      OP_REQUIRES_OK(ctx, ctx->input("read", &reads));

      SnapReadDecode read_batch(reads);
      size_t num_reads = read_batch.size();

      vector<Read*> input_reads;
      input_reads.reserve(num_reads);

      for (size_t i = 0; i < num_reads; i++) {
        if (read_batch.bases_len(i) == 0) {
          //LOG(INFO) << "string was empty, is this a partial batch?";
          continue;
        }

        Read* snap_read = new Read();
        snap_read->init(
            read_batch.metadata(i),  // id
            read_batch.metadata_len(i), // id len
            read_batch.bases(i),  // data (bases)
            read_batch.qualities(i),  // qualities
            read_batch.bases_len(i)  // data len
            );

        input_reads.push_back(snap_read);
      }

      size_t num_actual_reads = input_reads.size();
      vector<SingleAlignmentResult> alignment_results;

      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, 
            SnapResultsDecode::get_results_shape(num_reads, 
              num_secondary_alignments_), &out));

      MutableSnapResultsDecode results(out);

      OP_REQUIRES_OK(ctx, ctx->set_output("reads_out", *reads));
      //forward_ref_input_to_ref_output(0, 1);

      bool first_is_primary;
      for (size_t i = 0; i < num_actual_reads; i++) {
        Status status = snap_wrapper::alignSingle(base_aligner_, options_resource_->value(), input_reads[i],
            &alignment_results, num_secondary_alignments_, first_is_primary);

        results.set_first_is_primary(i, first_is_primary);
        results.set_num_results(i, alignment_results.size()); 
        for (int j = 0; j < alignment_results.size(); j++) {
          SingleAlignmentResult result = alignment_results[i];
          results.set_result_type(i, j, (int64)result.status); // cast from enum
          results.set_genome_location(i, j, GenomeLocationAsInt64(result.location));
          results.set_score(i, j, result.score);
          results.set_mapq(i, j, result.mapq);
          results.set_direction(i, j, result.direction);
        }
        alignment_results.clear();

        // TODO(solal): Smart pointers would probably make this unnecessary, but I'm not knowledgeable enough in C++ to do that
        delete input_reads[i];

        if (!status.ok()) {
          LOG(INFO) << "SnapAlignOp: alignSingle failed!!";
        }
      }

      //LOG(INFO) << "actual: " << num_actual_reads << " total: " << num_reads;
      for (size_t i = num_actual_reads; i < num_reads; i++) {
        // for uneven batches, set the num of results to 0
        results.set_num_results(i, 0);
      }
    }

  private:
    BaseAligner* base_aligner_ = nullptr;
    int num_secondary_alignments_ = 0;
    GenomeIndexResource* index_resource_ = nullptr;
    AlignerOptionsResource* options_resource_ = nullptr;

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
