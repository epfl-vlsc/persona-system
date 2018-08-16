// Modified version of kernels/sort/agd_merge_op.cc to sort proteins based on their sequence length
// Milad, Aug. 2018

#include <vector>
#include <memory>
#include <queue>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"

#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"


namespace tensorflow {
    using namespace std;
    using namespace errors;
    using namespace format;

    bool sort_ascending;

    namespace {
        //const string op_name("AGDProteinMerge");

        string PrintNormalizedProtein(const char* seq, size_t len) {
            vector<char> scratch;
            scratch.resize(len);
            memcpy(&scratch[0], seq, len);
            for (size_t i = 0; i < len; i++) {
                scratch[i] = scratch[i] + 'A';
            }
            return string(&scratch[0], len);
        }

        class ColumnCursor {
        public:
            ColumnCursor(AGDRecordReader &&proteins, vector<AGDRecordReader> &&other_columns) :
                    proteins_(move(proteins)), other_columns_(move(other_columns)) {}

            Status set_current_protein_size() {
                const char* data;
                size_t data_sz;
                TF_RETURN_IF_ERROR(proteins_.PeekNextRecord(&data, &data_sz));

                current_protein_size = data_sz;
                return Status::OK();
            }

            Status append_to_buffer_pairs(vector<BufferPair*> &bp_vec) {
                // first, dump the alignment result in the first column
                auto bp_proteins = bp_vec[0];
                TF_RETURN_IF_ERROR(copy_record(bp_proteins, proteins_));
                size_t bl_idx = 1;
                for (auto &r : other_columns_) {
                    auto bp = bp_vec[bl_idx++];
                    TF_RETURN_IF_ERROR(copy_record(bp, r));
                }
                return Status::OK();
            }

            inline size_t get_current_protein_size() {
                return current_protein_size;
            }

        private:
            vector<AGDRecordReader> other_columns_;
            AGDRecordReader proteins_;
            size_t current_protein_size;
            static inline
            Status
            copy_record(BufferPair* bp, AGDRecordReader &r) {
                const char *record_data;
                size_t record_size;
                auto &index = bp->index();
                auto &data = bp->data();

                TF_RETURN_IF_ERROR(r.GetNextRecord(&record_data, &record_size));

                RelativeIndex idx_sz = static_cast<RelativeIndex>(record_size);
                TF_RETURN_IF_ERROR(index.AppendBuffer(reinterpret_cast<const char*>(&idx_sz), sizeof(idx_sz))); // is it ok? mult*4?
                TF_RETURN_IF_ERROR(data.AppendBuffer(record_data, record_size));

                return Status::OK();
            }
        };



        typedef pair<size_t, ColumnCursor*> ProteinScore;
        struct ScoreComparator {
            bool operator()(const ProteinScore &a, const ProteinScore &b) {
                if (sort_ascending)
                    return a.first > b.first;
                return a.first < b.first;
            }
        };
    }

    class AGDProteinMergeOp : public OpKernel {
    public:
        AGDProteinMergeOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
            OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
            OP_REQUIRES_OK(ctx, ctx->GetAttr("sort_ascending", &sort_ascending_));
            sort_ascending = sort_ascending_;
        }

        ~AGDProteinMergeOp() {
            core::ScopedUnref queue_unref(queue_);
        }

        void Compute(OpKernelContext* ctx) override {
            if (!queue_) {
                OP_REQUIRES_OK(ctx, Init(ctx));
            }

            const Tensor *chunk_group_handles_t;
            OP_REQUIRES_OK(ctx, ctx->input("chunk_group_handles", &chunk_group_handles_t));
            auto chunk_group_shape = chunk_group_handles_t->shape();
            auto num_super_chunks = chunk_group_shape.dim_size(0);
            auto num_columns = chunk_group_shape.dim_size(1);
            auto chunk_group_handles = chunk_group_handles_t->tensor<string, 3>();

            auto rsrc_mgr = ctx->resource_manager();

            vector<ColumnCursor> columns;
            vector<unique_ptr<ResourceContainer<Data>, decltype(DataResourceReleaser)&>> releasers;

            // Note: we don't keep the actual ColumnCursors in here. all the move and copy ops would get expensive!

            priority_queue<ProteinScore, vector<ProteinScore>, ScoreComparator> score_heap;

            releasers.reserve(num_super_chunks * num_columns);
            columns.reserve(num_super_chunks);
            bool success = false;
            ResourceContainer<Data> *data;

            decltype(num_columns) column;
            LOG(INFO) << "AGD Merge: merging " << num_super_chunks << " intermediate files ...";
            for (decltype(num_super_chunks) super_chunk = 0; super_chunk < num_super_chunks; ++super_chunk) {
                column = 0;
                // First, we look up the results column
                OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(chunk_group_handles(super_chunk, column, 0),
                                                     chunk_group_handles(super_chunk, column, 1), &data));
                AGDRecordReader proteins_column { AGDRecordReader::fromUncompressed(data, &success) };
                OP_REQUIRES(ctx, success, Internal("Unable to parse results column fromUncompressed for Merge"));
                //LOG(INFO) << "chunk has " << proteins_column.NumRecords() << " records";
                releasers.push_back(move(decltype(releasers)::value_type(data, DataResourceReleaser)));

                // Then we look up the rest of the columns
                vector<AGDRecordReader> other_columns;
                other_columns.reserve(num_columns-1);
                for (column = 1; column < num_columns; ++column) {
                    OP_REQUIRES_OK(ctx, rsrc_mgr->Lookup(chunk_group_handles(super_chunk, column, 0),
                                                         chunk_group_handles(super_chunk, column, 1), &data));
                    AGDRecordReader other_column { AGDRecordReader::fromUncompressed(data, &success) };
                    OP_REQUIRES(ctx, success, Internal("Unable to parse other column fromUncompressed for Merge"));
                    other_columns.push_back(move(other_column));
                    releasers.push_back(move(decltype(releasers)::value_type(data, DataResourceReleaser)));
                }

                ColumnCursor a(move(proteins_column), move(other_columns));
                OP_REQUIRES_OK(ctx, a.set_current_protein_size());
                columns.push_back(move(a));
            }

            // Now that everything is initialized, add the scores to the heap
            for (auto &cc : columns) { // initial capacity of columns == num_super_chunks
                score_heap.push(ProteinScore(cc.get_current_protein_size(), &cc));
            }

            int current_chunk_size = 0;
            ColumnCursor *cc;
            vector<ResourceContainer<BufferPair>*> bp_ctrs;
            vector<BufferPair*> bufferpairs;
            bp_ctrs.resize(num_columns);
            for (auto& bp : bp_ctrs) {
                OP_REQUIRES_OK(ctx, bufpair_pool_->GetResource(&bp));
                bp->get()->reset();
                bufferpairs.push_back(bp->get());
            }

            Status s;
            while (!score_heap.empty()) {

                auto &top = score_heap.top();
                cc = top.second;
                cc->append_to_buffer_pairs(bufferpairs);

                score_heap.pop();

                s = cc->set_current_protein_size();
                if (s.ok()) {
                    // get_location will have the location advanced by the append_to_buffer_list call above
                    score_heap.push(ProteinScore(cc->get_current_protein_size(), cc));
                } else if (!IsResourceExhausted(s)) {
                    OP_REQUIRES_OK(ctx, s);
                }

                // pre-increment because we just added 1 to the chunk size
                // we're guaranteed that chunk size is at least 1
                if (++current_chunk_size == chunk_size_) {
                    OP_REQUIRES_OK(ctx, EnqueueBufferPairs(ctx, bp_ctrs, current_chunk_size));
                    bufferpairs.clear();
                    for (auto& bp : bp_ctrs) {
                        OP_REQUIRES_OK(ctx, bufpair_pool_->GetResource(&bp));
                        bp->get()->reset();
                        bufferpairs.push_back(bp->get());
                    }
                    current_chunk_size = 0;
                }
            }

            if (current_chunk_size > 0) {
                OP_REQUIRES_OK(ctx, EnqueueBufferPairs(ctx, bp_ctrs, current_chunk_size));
            }
            // Not sure if needed when using a queue runner?
            //queue_->Close(ctx, false, [](){});
        }

    private:

        QueueInterface *queue_ = nullptr;
        ReferencePool<BufferPair> *bufpair_pool_ = nullptr;
        TensorShape enqueue_shape_{{2}}, num_records_shape_{};
        int chunk_size_;
        bool sort_ascending_;

        Status Init(OpKernelContext *ctx) {
            TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 1), &queue_));
            TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pair_pool", &bufpair_pool_));
            return Status::OK();
        }

        Status EnqueueBufferPairs(OpKernelContext *ctx, vector<ResourceContainer<BufferPair>*> &bp_ctrs, size_t chunk_size) {
            QueueInterface::Tuple tuple; // just a vector<Tensor>
            vector<Tensor> containers_out;
            Tensor num_recs_out;
            containers_out.resize(bp_ctrs.size());
            for (auto& t : containers_out)
                TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, enqueue_shape_, &t));

            TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, num_records_shape_, &num_recs_out));
            num_recs_out.scalar<int>()() = chunk_size;
            tuple.push_back(num_recs_out);

            for (size_t i = 0; i < bp_ctrs.size(); i++) {
                auto container_out_vec = containers_out[i].vec<string>();
                container_out_vec(0) = bp_ctrs[i]->container();
                container_out_vec(1) = bp_ctrs[i]->name();
                tuple.push_back(containers_out[i]); // performs a shallow copy. Destructor doesn't release resources
            }

            TF_RETURN_IF_ERROR(queue_->ValidateTuple(tuple));

            // This is the synchronous version
            Notification n;
            queue_->TryEnqueue(tuple, ctx, [&n]() { n.Notify(); });
            n.WaitForNotification();

            return Status::OK();
        }
    };

    //REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDProteinMergeOp);
    REGISTER_KERNEL_BUILDER(Name("AGDProteinMerge").Device(DEVICE_CPU), AGDProteinMergeOp);
} // namespace tensorflow {


