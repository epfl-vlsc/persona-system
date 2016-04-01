#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
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

            OP_REQUIRES_OK(context,
                           GetResourceFromContext(context, "options_handle", &options_resource));
            OP_REQUIRES_OK(context,
                           GetResourceFromContext(context, "genome_handle", &index_resource));

            if (base_aligner_ == nullptr) {
                LOG(INFO) << "SNAP Kernel creating BaseAligner";
        
                base_aligner_ = snap_wrapper::createAligner(index_resource->value(), options_resource->value());
            }


            const Tensor* reads;
            OP_REQUIRES_OK(ctx, ctx->input("read", &reads));
            
            auto reads_flat = reads->flat<string>();
            size_t num_reads = reads_flat.size();

            vector<Read*> input_reads;
            vector<SnapProto::AlignmentDef> alignments;
            alignments.reserve(num_reads);
            input_reads.reserve(num_reads);

            for (size_t i = 0; i < num_reads; i++) {
                SnapProto::AlignmentDef alignment;
                SnapProto::ReadDef read_proto;
                if (!alignment.ParseFromString(reads_flat(i))) {
                    LOG(INFO) << "SnapAlignOp: failed to parse read from protobuf";
                }

                read_proto = alignment.read();
                Read* snap_read = new Read();
                snap_read->init(
                    read_proto.meta().c_str(),
                    read_proto.meta().length(),
                    read_proto.bases().c_str(),
                    read_proto.qualities().c_str(),
                    read_proto.length()
                );

                input_reads.push_back(snap_read);
                alignments.push_back(alignment);
                LOG(INFO) << "SnapAlignOp: added read " << read_proto.bases();
            }


            vector<vector<SingleAlignmentResult>> alignment_results;
            alignment_results.reserve(num_reads);

            for (size_t i = 0; i < input_reads.size(); i++) {
                Status status = snap_wrapper::alignSingle(base_aligner_, options_resouce->value(), input_reads[i], &alignment_results[i]);

                // TODO(solal): Smart pointers would probably make this unnecessary, but I'm not knowledgeable enough in C++ to do that
                delete input_reads[i];

                if (!status.ok()) {
                    LOG(INFO) << "SnapAlignOp: alignSingle failed!!";
                }
            }

            // shape of output tensor is [num_reads] 
            Tensor* out = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({num_reads}), &out));

            auto out_t = out->flat<string>();
            for (size_t i = 0; i < num_reads; i++) {

                SnapProto::AlignmentDef* alignment = &alignments[i];

                for (auto result : alignment_results[i]) {
                    SnapProto::SingleResult* result_proto = alignment.add_results();
                    populateSingleResultProto_(result_proto, result);
                }

                alignment->SerializeToString(&out_t(i));
            }
        }

    private:
        void populateSingleResultProto_(SnapProto::SingleResult* result_proto, SingleAlignmentResult result) {
            result_proto->set_result((SnapProto::SingleResult::AlignmentResult)result.status);
            result_proto->set_genomelocation(GenomeLocationAsInt64(result.location));
            result_proto->set_score(result.score);
            result_proto->set_mapq(result.mapq);
            result_proto->set_direction(result.direction);
        }

        BaseAligner* base_aligner_ = nullptr;
        
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
