#include <vector>
#include <tuple>
#include <thread>
#include <chrono>
#include <atomic>
#include <locale>
#include <pthread.h>
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

// uncommenting this will define the old behavior. may need to adjust / fix compile bugs
// #define YIELDING

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
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
#ifdef YIELDING
      OP_REQUIRES_OK(ctx, ctx->GetAttr("threads", &threads_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_yielding_threads", &num_yielding_threads_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("low_watermark", &low_watermark_));
#endif
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
      int i;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("trace_granularity", &i));
      OP_REQUIRES(ctx, i > 0, errors::InvalidArgument("trace granularity ", i, " must be greater than 0"));
      trace_granularity_ = i;

#ifdef YIELDING
      num_threads_ = threads_.size();
#else
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
#endif

      int capacity = (chunk_size_ / subchunk_size_) + 1;
      request_queue_.reset(new WorkQueue<tuple<ReadResource*, Buffer*, decltype(id_)>>(capacity));
      compute_status_ = Status::OK();
    }

    ~SnapAlignDenseParallelOp() override {
      if (!run_) {
        LOG(ERROR) << "Unable to safely wait in ~SnapAlignDenseParallelOp for all threads. run_ was toggled to false\n";
      }
      run_ = false;
      request_queue_->unblock();
      core::ScopedUnref index_unref(index_resource_);
      core::ScopedUnref options_unref(options_resource_);
      core::ScopedUnref buflist_pool_unref(buflist_pool_);
      while (num_active_threads_.load() > 0) {
        this_thread::sleep_for(chrono::milliseconds(10));
      }
      VLOG(INFO) << "request queue push wait: " << request_queue_->num_push_waits();
      VLOG(INFO) << "request queue pop wait: " << request_queue_->num_pop_waits();
      VLOG(DEBUG) << "Dense Align Destructor(" << this << ") finished\n";
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
    if (index_resource_ == nullptr) {
      OP_REQUIRES_OK(ctx, InitHandles(ctx));
      init_workers(ctx);
    }

    if (!compute_status_.ok()) {
      ctx->SetStatus(compute_status_);
      return;
    }

    OP_REQUIRES(ctx, run_, Internal("One of the aligner threads triggered a shutdown of the aligners. Please inspect!"));
    kernel_start = clock();

    ResourceContainer<ReadResource> *reads_container;
    const Tensor *read_input;
    OP_REQUIRES_OK(ctx, ctx->input("read", &read_input));
    auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &reads_container));
    core::ScopedUnref a(reads_container);
    auto reads = reads_container->get();
    tracepoint(bioflow, assembled_ready_queue_stop, reads_container);

    ResourceContainer<BufferList> *bufferlist_resource_container;
    OP_REQUIRES_OK(ctx, GetResultBufferList(ctx, &bufferlist_resource_container));
    auto alignment_result_buffer_list = bufferlist_resource_container->get();


    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, read_resources_));
    decltype(read_resources_)::size_type num_subchunks = read_resources_.size();
    alignment_result_buffer_list->resize(num_subchunks);

    for (decltype(num_subchunks) i = 0; i < num_subchunks; ++i) {
      auto *alignment_buffer = alignment_result_buffer_list->get_at(i);
      alignment_buffer->reset();
      tracepoint(bioflow, align_ready_queue_start, alignment_buffer);
      request_queue_->push(make_tuple(read_resources_[i].get(), alignment_buffer, id_));
    }
    tracepoint(bioflow, total_align_start, bufferlist_resource_container);
    pending_resources_.push_back(ReadResourceHolder(id_++, reads, num_subchunks));

    // needed because if we just call clear, this will
    // call delete on all the resources!
    for (auto &read_rsrc : read_resources_) {
      read_rsrc.release();
    }
    read_resources_.clear();
    tracepoint(bioflow, snap_align_kernel, kernel_start, reads_container);
    tracepoint(bioflow, result_ready_queue_start, bufferlist_resource_container);
  }

private:
  clock_t kernel_start;

  inline void init_workers(OpKernelContext* ctx) {
    auto aligner_func = [this] () {
      int my_id = 0;
      {
        mutex_lock l(mu_);
        my_id = thread_id_++;
      }
#ifdef YIELDING
      bool should_yield = my_id < num_yielding_threads_;
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(threads_[my_id], &cpuset);
      int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
      if (rc != 0) {
        LOG(INFO) << "Error calling pthread_setaffinity_np: " << rc << ", to core: " << threads_[my_id] 
          << " for thread id: " << my_id;
      }
#endif

      int capacity = request_queue_->capacity();
      //LOG(INFO) << "aligner thread spinning up";
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
      uint32_t num_completed;
      const uint32_t max_completed = trace_granularity_;
      while (run_) {
        if (request_queue_->pop(batch)) {
          reads = get<0>(batch);
          result_buf = get<1>(batch);
          tracepoint(bioflow, subchunk_time_start, result_buf);
          tracepoint(bioflow, align_ready_queue_stop, result_buf);
          id = get<2>(batch);
#ifdef YIELDING
          if (should_yield && (float)request_queue_->size() / (float)capacity < low_watermark_)
            std::this_thread::yield();
#endif
        } else
          continue;

        // should this by in the if statement above?
        num_completed = 0;

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
              result_builder.AppendAlignmentResult(primaryResult, "*", 4, res_buf);
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
                                                 &snap_read, primaryResult, first_is_primary, format,
                                                 options_->useM, lvc, genome_, cigarString, flag);

          if (!s.ok())
            LOG(ERROR) << "computeCigarFlags did not return OK!!!";

          /*LOG(INFO) << " result: location " << primaryResult.location <<
            " direction: " << primaryResult.direction << " score " << primaryResult.score << " cigar: " << cigarString << " mapq: " << primaryResult.mapq;*/

          result_builder.AppendAlignmentResult(primaryResult, cigarString, flag, res_buf);
          if (++num_completed == max_completed) {
            tracepoint(bioflow, reads_aligned, num_completed);
            num_completed = 0;
          }
        }

        if (num_completed > 0) {
          tracepoint(bioflow, reads_aligned, num_completed);
        }

        if (!IsResourceExhausted(status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError! : " << status << "\n";
          compute_status_ = status;
          return;
        }

        result_builder.AppendAndFlush(res_buf);
        result_buf->set_ready();
        tracepoint(bioflow, subchunk_time_stop, result_buf);
      }

      VLOG(INFO) << "base aligner thread ending.";
      num_active_threads_--;
    };
    auto worker_threadpool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    for (int i = 0; i < num_threads_; i++)
      worker_threadpool->Schedule(aligner_func);
    num_active_threads_ = num_threads_;
  }

  ReferencePool<BufferList> *buflist_pool_ = nullptr;
  GenomeIndexResource* index_resource_ = nullptr;
  AlignerOptionsResource* options_resource_ = nullptr;
  const Genome *genome_ = nullptr;
  AlignerOptions* options_ = nullptr;
  int subchunk_size_;
  int chunk_size_;
  volatile bool run_ = true;
  uint64_t id_ = 0;
#ifdef YIELDING
  float low_watermark_;
  int num_yielding_threads_;
  std::vector<int> threads_;
#endif
  uint32_t trace_granularity_;

  atomic<uint32_t> num_active_threads_;
  mutex mu_;
  int thread_id_ = 0;

  int num_threads_, num_yielding_threads_;
  unique_ptr<WorkQueue<tuple<ReadResource*, Buffer*, decltype(id_)>>> request_queue_;

  vector<uint64_t> completion_process_queue_;
  vector<unique_ptr<ReadResource>> read_resources_;
  vector<ReadResourceHolder> pending_resources_;
  Status compute_status_;
  TF_DISALLOW_COPY_AND_ASSIGN(SnapAlignDenseParallelOp);
};

  REGISTER_OP("SnapAlignDenseParallel")
#ifdef YIELDING
  .Attr("threads: list(int)")
  .Attr("num_yielding_threads: int")
  .Attr("low_watermark: float")
#else
  .Attr("num_threads: int")
#endif
  .Attr("trace_granularity: int = 500")
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
