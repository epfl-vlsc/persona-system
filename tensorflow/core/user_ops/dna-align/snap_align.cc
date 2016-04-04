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
#include "aligner_options_resource.h"

namespace tensorflow {
    using namespace std;

    class SnapAlignOp : public OpKernel {
    public:
        explicit SnapAlignOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

        ~SnapAlignOp() override {
            //    if (genome_index_) genome_index_->Unref();
        }

        void Compute(OpKernelContext* ctx) override {

            GenomeIndexResource* index_resource;
            AlignerOptionsResource* options_resource;

            OP_REQUIRES_OK(ctx,
                GetResourceFromContext(ctx, "options_handle", &options_resource));
            OP_REQUIRES_OK(ctx,
                GetResourceFromContext(ctx, "genome_handle", &index_resource));

            OP_REQUIRES_OK(ctx,
                snap_wrapper::init());

            if (base_aligner_ == nullptr) {
                LOG(INFO) << "SNAP Kernel creating BaseAligner";

                if (index_resource == nullptr) {
                    LOG(INFO) << "INDEX IS NULL";
                }
                if (index_resource->get_index() == nullptr) {
                    LOG(INFO) << "INDEX VALUE IS NULL";
                }
                if (options_resource == nullptr) {
                    LOG(INFO) << "OPTIONS IS NULL";
                }
                if (options_resource->value() == nullptr) {
                    LOG(INFO) << "OPTIONS VALUE IS NULL";
                }

                base_aligner_ = snap_wrapper::createAligner(index_resource->get_index(), options_resource->value());

                LOG(INFO) << "1";

                AlignerOptions* options = options_resource->value();

                LOG(INFO) << "2";

                if (options->maxSecondaryAlignmentAdditionalEditDistance < 0) {
                    LOG(INFO) << "3";
                    num_secondary_alignments_ = 0;
                }
                else {
                    LOG(INFO) << "4";
                    num_secondary_alignments_ = BaseAligner::getMaxSecondaryResults(options->numSeedsFromCommandLine,
                        options->seedCoverage, MAX_READ_LENGTH, options->maxHits, index_resource->get_index()->getSeedLength());
                }
            }

            LOG(INFO) << "5";

            const Tensor* reads;
            OP_REQUIRES_OK(ctx, ctx->input("read", &reads));

            LOG(INFO) << "6";

            auto reads_flat = reads->flat<string>();
            size_t num_reads = reads_flat.size();

            LOG(INFO) << "7";

            vector<Read*> input_reads;
            vector<SnapProto::AlignmentDef> alignments;
            alignments.reserve(num_reads);
            input_reads.reserve(num_reads);

            LOG(INFO) << "8";

            for (size_t i = 0; i < num_reads; i++) {
                SnapProto::AlignmentDef alignment;
                SnapProto::ReadDef read_proto;
                if (!alignment.ParseFromString(reads_flat(i))) {
                    LOG(INFO) << "SnapAlignOp: failed to parse read from protobuf";
                }

                LOG(INFO) << "8_" << i;

                read_proto = alignment.read();
                Read* snap_read = new Read();
                snap_read->init(
                    read_proto.meta().c_str(),
                    read_proto.meta().length(),
                    read_proto.bases().c_str(),
                    read_proto.qualities().c_str(),
                    read_proto.length()
                );

                LOG(INFO) << "9_" << i;

                input_reads.push_back(snap_read);
                alignments.push_back(alignment);
                LOG(INFO) << "SnapAlignOp: added read " << read_proto.bases();
            }

            LOG(INFO) << "10";

            vector<vector<SingleAlignmentResult>> alignment_results;
            alignment_results.reserve(num_reads);

            for (size_t i = 0; i < input_reads.size(); i++) {
                Status status = snap_wrapper::alignSingle(base_aligner_, options_resource->value(), input_reads[i],
                    &alignment_results[i], num_secondary_alignments_);

                LOG(INFO) << "11";

                // TODO(solal): Smart pointers would probably make this unnecessary, but I'm not knowledgeable enough in C++ to do that
                delete input_reads[i];

                LOG(INFO) << "12";

                if (!status.ok()) {
                    LOG(INFO) << "SnapAlignOp: alignSingle failed!!";
                }
            }

            LOG(INFO) << "13";

            // shape of output tensor is [num_reads] 
            Tensor* out = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ (int64)num_reads }), &out));

            LOG(INFO) << "14";

            auto out_t = out->flat<string>();
            for (size_t i = 0; i < num_reads; i++) {
                LOG(INFO) << "15_" << i;

                SnapProto::AlignmentDef* alignment = &alignments[i];
                if (alignment == nullptr) {
                    LOG(INFO) << "ALIGNMENT IS NULL";
                }

                LOG(INFO) << "16_" << i;

                int counter = 0;
                for (auto result : alignment_results[i]) {
                    LOG(INFO) << "16_1_" << i << "_" << counter;
                    SnapProto::SingleResultDef* result_proto = alignment->add_results();
                    if (result_proto == nullptr) {
                        LOG(INFO) << "RESULT IS NULL";
                    }
                    LOG(INFO) << "16_2_" << i << "_" << counter;
                    populateSingleResultProto_(result_proto, result);

                    counter++;
                }
                LOG(INFO) << "17_" << i;

                alignment->SerializeToString(&out_t(i));
            }

            LOG(INFO) << "18";
        }

    private:
        void populateSingleResultProto_(SnapProto::SingleResultDef* result_proto, SingleAlignmentResult result) {
            result_proto->set_result((SnapProto::SingleResultDef::AlignmentResult)result.status);
            result_proto->set_genomelocation(GenomeLocationAsInt64(result.location));
            result_proto->set_score(result.score);
            result_proto->set_mapq(result.mapq);
            result_proto->set_direction(result.direction);
        }

        BaseAligner* base_aligner_ = nullptr;
        int num_secondary_alignments_ = 0;

    };


    REGISTER_OP("SnapAlign")
        .Input("genome_handle: Ref(string)")
        .Input("options_handle: Ref(string)")
        .Input("read: string")
        .Output("output: string")
        .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
output: a tensor [num_reads] containing serialized reads and results
containing the alignment candidates. 
)doc");


    REGISTER_KERNEL_BUILDER(Name("SnapAlign").Device(DEVICE_CPU), SnapAlignOp);

}  // namespace tensorflow
