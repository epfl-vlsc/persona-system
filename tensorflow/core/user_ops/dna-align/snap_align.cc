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

            options_ = new snap_wrapper::AlignmentOptions();
            // TODO configure options here
        }

        ~SnapAlignOp() override {
            if (genome_index_) genome_index_->Unref();
        }

        void Compute(OpKernelContext* ctx) override {
            {
                mutex_lock l(init_mu_);

                if (genome_index_ == nullptr) {
                    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def()));

                    auto creator = [this](GenomeIndexResource** index) {
                        *index = new GenomeIndexResource();
                        (*index)->init(genome_index_location_);
                        return Status::OK();
                    };

                    OP_REQUIRES_OK(ctx, cinfo_.resource_manager()->LookupOrCreate<GenomeIndexResource>(
                        cinfo_.container(), cinfo_.name(), &genome_index_, creator));
            
                    base_aligner_ = snap_wrapper::createAligner(genome_index_->value(), options_);
                }
            }

            const Tensor* reads;

            // get the input tensor called "read"
            OP_REQUIRES_OK(ctx, ctx->input("read", &reads));
            
            auto reads_t = reads->flat<string>();
            unsigned int num_reads = reads_t.size();
            vector<Read*> input_reads;
            input_reads.reserve(num_reads);
            SnapProto::Read read_proto;

            for (int i = 0; i < num_reads; i++) {
                if (!read_proto.ParseFromString(reads_t(i)))
                    LOG(INFO) << "SNAP Align: failed to parse read from protobuf";

                Read* snap_read = new Read();
                snap_read.init(read_proto.meta().c_str(), read_protosmeta().length(), read_protosbases().c_str(),
                        read_proto.qualities().c_str(), read_proto.length());

                input_reads.push_back(snap_read);

                LOG(INFO) << "SnapAlignOp: added read " << read_proto.bases();
            }


            auto alignment_results = new vector<vector<SingleAlignmentResult*>>();
            alignment_results->reserve(num_reads);

            // call align() here for each input_reads[i]
            for (int i = 0; i < input_reads.size(); i++) {
                // call align
                vector<SingleAlignmentResult*>* results = &(*alignment_results)[i]; // a little messy :-/
                Status status = snap_wrapper::alignSingle(base_aligner_, options_, input_reads[i], results);
                if (!status.ok())
                    LOG(INFO) << "alignSingle failed!!";
            }
            
            // shape of output tensor is [num_reads, 2] 
            Tensor* out = nullptr;
            OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_reads, 2}), &out));

            auto out_t = out->matrix<string>();

            for (int i = 0; i < num_reads; i++) {
                out_t(i, 0) = reads_t(i); // copy over the read (not sure if uses same buffer)

                SnapProto::AlignmentResults results;

                for (auto result : alignment_results[i]) {
                    SnapProto::SingleResult* result_proto = results.add_results();
                    populateSingleResultProto(result_proto, result);
                }

                results.SerializeToString(&out_t(i, 1));
            }

        }

    private:

        void populateSingleResultProto_(SnapProto::SingleResult* result_proto, SingleAlignmentResult& result) {
            result_proto->set_result(result.status);
            result_proto->set_genomeLocation(GenomeLocationAsInt64(result.location));
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

        // Options
        snap_wrapper::AlignmentOptions* options_;
        // Attributes
        string genome_index_location_;

        mutex init_mu_;
        ContainerInfo cinfo_ GUARDED_BY(init_mu_);
        GenomeIndexResource* genome_index_ GUARDED_BY(init_mu_) = nullptr;
        BaseAligner* base_aligner_ = nullptr;
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
output: a tensor [num_reads, 2] containing serialized reads and results
containing the alignment candidates. 
)doc");


    REGISTER_KERNEL_BUILDER(Name("SnapAlign").Device(DEVICE_CPU), SnapAlignOp);

}  // namespace tensorflow
