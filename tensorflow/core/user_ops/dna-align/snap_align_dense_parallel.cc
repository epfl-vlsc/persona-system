#include <sys/time.h>
#include <sys/resource.h>
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
#include "tensorflow/core/user_ops/agd-format/buffer_list.h"
#include "tensorflow/core/user_ops/agd-format/column_builder.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/FileFormat.h"
#include "tensorflow/core/user_ops/agd-format/column_builder.h"
#include "GenomeIndex.h"
#include "work_queue.h"
#include "Read.h"
#include "SnapAlignerWrapper.h"
#include "genome_index_resource.h"
#include "aligner_options_resource.h"
#include "tensorflow/core/user_ops/agd-format/read_resource.h"
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

class SnapAlignAGDParallelOp : public OpKernel {
  public:
    explicit SnapAlignAGDParallelOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
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
      request_queue_.reset(new WorkQueue<tuple<ReadResource*, Buffer*, decltype(id_), ReadResource*>>(capacity));
      compute_status_ = Status::OK();
    }

    ~SnapAlignAGDParallelOp() override {
      if (!run_) {
        LOG(ERROR) << "Unable to safely wait in ~SnapAlignAGDParallelOp for all threads. run_ was toggled to false\n";
      }
      run_ = false;
      request_queue_->unblock();
      core::ScopedUnref index_unref(index_resource_);
      core::ScopedUnref options_unref(options_resource_);
      core::ScopedUnref buflist_pool_unref(buflist_pool_);
      while (num_active_threads_.load() > 0) {
        this_thread::sleep_for(chrono::milliseconds(10));
      }
      LOG(INFO) << "request queue push wait: " << request_queue_->num_push_waits();
      LOG(INFO) << "request queue pop wait: " << request_queue_->num_pop_waits();
      //LOG(INFO) << "done queue push wait: " << done_queue_->num_push_waits();
      //LOG(INFO) << "done queue pop wait: " << done_queue_->num_pop_waits();
      VLOG(DEBUG) << "AGD Align Destructor(" << this << ") finished\n";
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
      request_queue_->push(make_tuple(read_resources_[i].get(), alignment_buffer, id_, reads));
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
      std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
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
      auto index = index_resource_->get_index();
      auto options = options_resource_->value();

      unsigned alignmentResultBufferCount;
      if (options->maxSecondaryAlignmentAdditionalEditDistance < 0) {
          alignmentResultBufferCount = 1; // For the primary alignment
      } else {
          alignmentResultBufferCount = BaseAligner::getMaxSecondaryResults(options->numSeedsFromCommandLine, options->seedCoverage, 
              MAX_READ_LENGTH, options->maxHits, index->getSeedLength()) + 1; // +1 for the primary alignment
      }
      size_t alignmentResultBufferSize = sizeof(SingleAlignmentResult) * (alignmentResultBufferCount + 1); // +1 is for primary result
   
      BigAllocator *allocator = new BigAllocator(BaseAligner::getBigAllocatorReservation(index, true, 
            options->maxHits, MAX_READ_LENGTH, index->getSeedLength(), options->numSeedsFromCommandLine, options->seedCoverage, options->maxSecondaryAlignmentsPerContig) 
          + alignmentResultBufferSize);
    
      LOG(INFO) << "reservation: " << BaseAligner::getBigAllocatorReservation(index, true, 
            options->maxHits, MAX_READ_LENGTH, index->getSeedLength(), options->numSeedsFromCommandLine, options->seedCoverage, options->maxSecondaryAlignmentsPerContig) 
          + alignmentResultBufferSize;

      BaseAligner* base_aligner = new (allocator) BaseAligner(
        index,
        options->maxHits,
        options->maxDist,
        MAX_READ_LENGTH,
        options->numSeedsFromCommandLine,
        options->seedCoverage,
        options->minWeightToCheck,
        options->extraSearchDepth,
        false, false, false, // stuff that would decrease performance without impacting quality
        options->maxSecondaryAlignmentsPerContig,
        nullptr, nullptr, // Uncached Landau-Vishkin
        nullptr, // No need for stats
        allocator 
        );
      
      allocator->checkCanaries();

      base_aligner->setExplorePopularSeeds(options->explorePopularSeeds);
      base_aligner->setStopOnFirstHit(options->stopOnFirstHit);
      
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
      ReadResource* reads, *parent_reads;
      decltype(id_) id;
      tuple<ReadResource*, Buffer*, decltype(id_), ReadResource*> batch;
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
          parent_reads = get<3>(batch);
#ifdef YIELDING
          if (should_yield && (float)request_queue_->size() / (float)capacity < low_watermark_)
            std::this_thread::yield();
#endif
        } else
          continue;

        // should this by in the if statement above?
        num_completed = 0;
        SubchunkReleaser a(*parent_reads);

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
            }
            continue;
          }


          base_aligner->AlignRead(
                                  &snap_read,
                                  &primaryResult,
                                  options_->maxSecondaryAlignmentAdditionalEditDistance,
                                  0, //num_secondary_alignments * sizeof(SingleAlignmentResult),
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
            tracepoint(bioflow, reads_aligned, num_completed, my_id, this);
            num_completed = 0;
          }
        }

        if (num_completed > 0) {
          tracepoint(bioflow, reads_aligned, num_completed, my_id, this);
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

      std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> thread_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
      struct rusage usage;
      int ret = getrusage(RUSAGE_THREAD, &usage);

      LOG(INFO) << "Aligner thread total time is: " << thread_time.count() << " seconds";
      LOG(INFO) << "system time used: " << usage.ru_stime.tv_sec << "." << usage.ru_stime.tv_usec << endl;
      LOG(INFO) << "user time used: " << usage.ru_utime.tv_sec << "." << usage.ru_utime.tv_usec << endl;
      LOG(INFO) << "maj page faults: " << usage.ru_minflt << endl;
      LOG(INFO) << "min page faults: " << usage.ru_majflt << endl;
      base_aligner->~BaseAligner(); // This calls the destructor without calling operator delete, allocator owns the memory.
      delete allocator;
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
  unique_ptr<WorkQueue<tuple<ReadResource*, Buffer*, decltype(id_), ReadResource*>>> request_queue_;

  // used to get things out of the queue quickly
  vector<decltype(id_)> completion_process_queue_;
  // TODO make this more efficient with a map from id_->read resource...
  vector<unique_ptr<ReadResource>> read_resources_;
  vector<ReadResourceHolder> pending_resources_;

  Status compute_status_;
  TF_DISALLOW_COPY_AND_ASSIGN(SnapAlignAGDParallelOp);
};

  REGISTER_OP("SnapAlignAGDParallel")
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

  REGISTER_KERNEL_BUILDER(Name("SnapAlignAGDParallel").Device(DEVICE_CPU), SnapAlignAGDParallelOp);

}  // namespace tensorflow
