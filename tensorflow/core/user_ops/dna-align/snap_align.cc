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

namespace tensorflow {
    using namespace std;

    class SnapAlignOp : public OpKernel {
    public:
        explicit SnapAlignOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("genome_index_location", &genome_index_location_));

            // TODO configure options here
        }

        ~SnapAlignOp() override {
            if (genome_index_) genome_index_->Unref();
        }

        void Compute(OpKernelContext* ctx) override {
            {
                mutex_lock l(init_mu_);

                if (genome_index_ == nullptr) {
                    LOG(INFO) << "SNAP Kernel creating/getting genome index and creating BaseAligner";
                    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def()));

                    auto creator = [this](GenomeIndexResource** index) {
                        *index = new GenomeIndexResource();
                        (*index)->init(genome_index_location_);
                        return Status::OK();
                    };

                    OP_REQUIRES_OK(ctx, 
                        cinfo_.resource_manager()->LookupOrCreate<GenomeIndexResource>(
                            cinfo_.container(),
                            cinfo_.name(),
                            &genome_index_,
                            creator
                        )
                    );
            
                    base_aligner_ = snap_wrapper::createAligner(genome_index_->value(), &options_);
                }
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
                Status status = snap_wrapper::alignSingle(base_aligner_, &options_, input_reads[i], &alignment_results[i]);

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

        class GenomeIndexResource : public ResourceBase {
        public:
            explicit GenomeIndexResource() {}

            GenomeIndex* value() { return value_; }

            void init(string path) {
                // 2nd and 3rd arguments are weird SNAP things that can safely be ignored
                value_ = GenomeIndex::loadFromDirectory(const_cast<char*>(path.c_str()), false, false);
            }

            string DebugString() override {
                return "SNAP GenomeIndex";
            }

        private:
            GenomeIndex* value_;

            TF_DISALLOW_COPY_AND_ASSIGN(GenomeIndexResource);
        };

        snap_wrapper::AlignmentOptions options_;
        BaseAligner* base_aligner_ = nullptr;

        // Attributes
        string genome_index_location_;

        // Resources
        mutex init_mu_;
        ContainerInfo cinfo_ GUARDED_BY(init_mu_);
        GenomeIndexResource* genome_index_ GUARDED_BY(init_mu_) = nullptr;
    };


    REGISTER_OP("SnapAlign")
        .Attr("genome_index_location: string")
        .Input("read: string")
        .Output("output: string")
        .Attr("container: string = 'gac_container'")
        .Attr("shared_name: string = 'gac_name'")
        .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
output: a tensor [num_reads] containing serialized reads and results
containing the alignment candidates. 
)doc");


    REGISTER_KERNEL_BUILDER(Name("SnapAlign").Device(DEVICE_CPU), SnapAlignOp);

}  // namespace tensorflow
