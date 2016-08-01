#include <vector>
#include <tuple>
#include <thread>
#include <chrono>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/dense-format/buffer_list.h"
#include "tensorflow/core/user_ops/dense-format/column_builder.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/FileFormat.h"
#include "tensorflow/core/user_ops/dense-format/column_builder.h"
#include "GenomeIndex.h"
#include "work_queue.h"
#include "Read.h"
#include "SnapAlignerWrapper.h"
#include "genome_index_resource.h"
#include "aligner_options_resource.h"
#include "tensorflow/core/user_ops/dense-format/read_resource.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"

namespace tensorflow {
using namespace std;
using namespace errors;

  namespace {
    class ReadResourceHolder
    {
    private:
      uint64_t id_;
      ReadResource *parent_resource_;
      uint16_t count_;

    public:
      ReadResourceHolder(const decltype(id_) id, ReadResource *parent_resource, uint16_t count) :
        id_(id), parent_resource_(parent_resource), count_(count) {}

      inline bool
      is_id(const decltype(id_) &other_id) {
        return other_id == id_;
      }

      inline bool
      decrement_count() {
        // complex if statement prevents wraparound, which shouldn't happen
        if (count_ == 0) {
          LOG(DEBUG) << "decrement_count called with count_ already == 0. This is a bug!\n";
          return true;
        } else if (--count_ == 0) {
          ReadResourceReleaser a(*parent_resource_);
          return true;
        } else
          return false;
      }

    };
  }

class SnapAlignDenseParallelOp : public OpKernel {
  public:
    explicit SnapAlignDenseParallelOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads",
              &num_threads_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size",
              &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size",
              &chunk_size_));
      int capacity = (chunk_size_ / subchunk_size_) + 1;
      request_queue_.reset(new WorkQueue<tuple<ReadResource*, Buffer*, decltype(id_)>>(capacity));
      completion_queue_.reset(new WorkQueue<uint64_t>(capacity));
    }

    ~SnapAlignDenseParallelOp() override {
      run_ = false;
      request_queue_->unblock();
      size_t requests_in_flight;
      while ((requests_in_flight = request_queue_->size()) > 0) {
        LOG(DEBUG) << "DenseAligner("<< this << ") waiting for " << requests_in_flight << " to finish\n";
        this_thread::sleep_for(chrono::milliseconds(250));
      }
      completion_queue_->unblock();
      core::ScopedUnref index_unref(index_resource_);
      core::ScopedUnref options_unref(options_resource_);
      core::ScopedUnref buflist_pool_unref(buflist_pool_);
      while (num_done_ != num_threads_);
      LOG(DEBUG) << "Dense Align Destructor finished\n";
    }

    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "options_handle", &options_resource_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "genome_handle", &index_resource_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
      TF_RETURN_IF_ERROR(snap_wrapper::init());

      options_ = options_resource_->value();
      genome_ = index_resource_->get_genome();

      /*if (options_->maxSecondaryAlignmentAdditionalEditDistance < 0) {
        num_secondary_alignments_ = 0;
      } else {
        num_secondary_alignments_ = BaseAligner::getMaxSecondaryResults(options_->numSeedsFromCommandLine,
            options_->seedCoverage, MAX_READ_LENGTH, options_->maxHits, index_resource_->get_index()->getSeedLength());
      }*/

      return Status::OK();
    }

    Status GetResultBufferList(OpKernelContext* ctx, ResourceContainer<BufferList> **ctr)
    {
      TF_RETURN_IF_ERROR(buflist_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      TF_RETURN_IF_ERROR((*ctr)->allocate_output("result_buf_handle", ctx));
      return Status::OK();
    }


  void Compute(OpKernelContext* ctx) override {
    auto start = clock();
    if (index_resource_ == nullptr) {
      OP_REQUIRES_OK(ctx, InitHandles(ctx));
      init_workers(ctx);
    }
    OP_REQUIRES(ctx, run_, Internal("One of the aligner threads triggered a shutdown of the aligners. Please inspect!"));

    ResourceContainer<BufferList> *bufferlist_resource_container;
    OP_REQUIRES_OK(ctx, GetResultBufferList(ctx, &bufferlist_resource_container));
    auto alignment_result_buffer_list = bufferlist_resource_container->get();

    ResourceContainer<ReadResource> *reads_container;
    const Tensor *read_input;
    OP_REQUIRES_OK(ctx, ctx->input("read", &read_input));
    auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &reads_container));
    core::ScopedUnref a(reads_container);
    auto reads = reads_container->get();

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, read_resources_));
    decltype(read_resources_)::size_type num_subchunks = read_resources_.size();
    alignment_result_buffer_list->resize(num_subchunks);

    for (decltype(num_subchunks) i = 0; i < num_subchunks; ++i) {
      request_queue_->push(make_tuple(read_resources_[i].get(), alignment_result_buffer_list->get_at(i), id_));
    }
    pending_resources_.push_back(ReadResourceHolder(id_++, reads, num_subchunks));

    OP_REQUIRES_OK(ctx, process_completed_chunks());

    // needed because if we just call clear, this will
    // call delete on all the resources!
    for (auto &read_rsrc : read_resources_) {
      read_rsrc.release();
    }
    read_resources_.clear();
    tracepoint(bioflow, snap_align_kernel, clock() - start);
  }

private:

  inline Status
  process_completed_chunks()
  {
    auto status = Status::OK();

    completion_queue_->pop_all(completion_process_queue_);
    bool found;
    for (auto &id : completion_process_queue_) {
      found = false;
      for (decltype(pending_resources_)::size_type i = 0; i < pending_resources_.size(); ++i) {
        auto &pending_resource = pending_resources_[i];
        if (pending_resource.is_id(id)) {
          if (pending_resource.decrement_count()) {
            pending_resources_.erase(pending_resources_.begin() + i);
          }
          found = true;
          break;
        }
      }
      if (!found) {
        status = Internal("Unable to find pending resource with id ", id);
        break;
      }
    }

    return status;
  }

  inline void init_workers(OpKernelContext* ctx) {
    auto aligner_func = [this] () {
      LOG(INFO) << "aligner thread spinning up";
      BaseAligner* base_aligner = snap_wrapper::createAligner(index_resource_->get_index(), options_resource_->value());
      bool first_is_primary = true; // we only ever generate one result
      const char *bases, *qualities;
      size_t bases_len, qualities_len;
      SingleAlignmentResult primaryResult;
      int num_secondary_alignments = 0;
      int num_secondary_results;
      SAMFormat format(options_->useM);
      AlignmentResultBuilder result_builder;
      string cigarString;
      int flag;
      Read snap_read;
      LandauVishkinWithCigar lvc;

      Buffer* result_buf;
      ReadResource* reads;
      decltype(id_) id;
      tuple<ReadResource*, Buffer*, decltype(id_)> batch;
      clock_t start;
      Status status;
      while (run_) {
        if (request_queue_->pop(batch)) {
          reads = get<0>(batch);
          result_buf = get<1>(batch);
          id = get<2>(batch);
        } else
          continue;

        start = clock();

        result_buf->reset();
        auto &res_buf = result_buf->get();
        for (status = reads->get_next_record(&bases, &bases_len, &qualities, &qualities_len);
             status.ok();
             status = reads->get_next_record(&bases, &bases_len, &qualities, &qualities_len)) {
          cigarString.clear();
          snap_read.init(nullptr, 0, bases, qualities, bases_len);
          snap_read.clip(options_->clipping);
          if (snap_read.getDataLength() < options_->minReadLength || snap_read.countOfNs() > options_->maxDist) {
            if (!options_->passFilter(&snap_read, AlignmentResult::NotFound, true, false)) {
              LOG(INFO) << "FILTERING READ";
            } else {
              primaryResult.status = AlignmentResult::NotFound;
              primaryResult.location = InvalidGenomeLocation;
              primaryResult.mapq = 0;
              primaryResult.direction = FORWARD;
              result_builder.AppendAlignmentResult(primaryResult, cigarString, 4, res_buf);
              continue;
            }
          }


          base_aligner->AlignRead(
                                  &snap_read,
                                  &primaryResult,
                                  options_->maxSecondaryAlignmentAdditionalEditDistance,
                                  num_secondary_alignments * sizeof(SingleAlignmentResult),
                                  &num_secondary_results,
                                  num_secondary_alignments,
                                  nullptr //secondaryResults
                                  );

          flag = 0;

          // we may need to do post process options->passfilter here?

          // compute the CIGAR strings and flags
          // input_reads[i] holds the current snap_read
          /*Status s = snap_wrapper::computeCigarFlags(
            &snap_read, &primaryResult, 1, first_is_primary, format,
            options_->useM, lvc, genome_, cigarString, flag);*/

          Status s = snap_wrapper::adjustResults(
                                                 &snap_read, &primaryResult, 1, first_is_primary, format,
                                                 options_->useM, lvc, genome_, cigarString, flag);

          if (!s.ok())
            LOG(INFO) << "computeCigarFlags did not return OK!!!";

          /*LOG(INFO) << " result: location " << primaryResult.location <<
            " direction: " << primaryResult.direction << " score " << primaryResult.score << " cigar: " << cigarString << " mapq: " << primaryResult.mapq;*/

          result_builder.AppendAlignmentResult(primaryResult, cigarString, flag, res_buf);
        }

        if (!IsResourceExhausted(status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError! : " << status << "\n";
          run_ = false; // not sure what to do here for errors
          return;
        }

        result_builder.AppendAndFlush(res_buf);
        result_buf->set_ready();
        if (run_)
          completion_queue_->push(id);
        tracepoint(bioflow, snap_alignments, clock() - start, reads->num_records());
      }

      LOG(INFO) << "base aligner thread ending.";
      mutex_lock l(done_mu_);
      num_done_++;
    };
    auto worker_threadpool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    for (int i = 0; i < num_threads_; i++)
      worker_threadpool->Schedule(aligner_func);
  }

  ReferencePool<BufferList> *buflist_pool_ = nullptr;
  GenomeIndexResource* index_resource_ = nullptr;
  AlignerOptionsResource* options_resource_ = nullptr;
  const Genome *genome_ = nullptr;
  AlignerOptions* options_ = nullptr;
  int num_threads_;
  int subchunk_size_;
  int chunk_size_;
  volatile bool run_ = true;
  uint64_t id_ = 0;
  mutex done_mu_;
  int num_done_ = 0;

  unique_ptr<WorkQueue<tuple<ReadResource*, Buffer*, decltype(id_)>>> request_queue_;
  unique_ptr<WorkQueue<uint64_t>> completion_queue_;

  vector<uint64_t> completion_process_queue_;
  vector<unique_ptr<ReadResource>> read_resources_;
  vector<ReadResourceHolder> pending_resources_;
  TF_DISALLOW_COPY_AND_ASSIGN(SnapAlignDenseParallelOp);
};

  REGISTER_OP("SnapAlignDenseParallel")
  .Attr("num_threads: int = 1")
  .Attr("chunk_size: int")
  .Attr("subchunk_size: int")
  .Input("genome_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
  .Input("buffer_list_pool: Ref(string)")
  .Input("read: string")
  .Output("result_buf_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
output: a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.
)doc");

  REGISTER_KERNEL_BUILDER(Name("SnapAlignDenseParallel").Device(DEVICE_CPU), SnapAlignDenseParallelOp);

}  // namespace tensorflow
