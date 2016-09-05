#include <sys/time.h>
#include <sys/resource.h>
#include <vector>
#include <tuple>
#include <thread>
#include <memory>
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

namespace tensorflow {
using namespace std;
using namespace errors;

  namespace {
    void resource_releaser(ResourceContainer<ReadResource> *rr) {
      ResourceReleaser<ReadResource> a(*rr);
      {
        ReadResourceReleaser r(*rr->get());
      }
    }
  }

class SnapAlignAGDParallelOp : public OpKernel {
  public:
    explicit SnapAlignAGDParallelOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
      int i;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("trace_granularity", &i));
      OP_REQUIRES(ctx, i > 0, errors::InvalidArgument("trace granularity ", i, " must be greater than 0"));
      trace_granularity_ = i;

      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));

      int capacity;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("work_queue_size", &capacity));
      request_queue_.reset(new WorkQueue<shared_ptr<ResourceContainer<ReadResource>>>(capacity));
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
      LOG(INFO) << "request queue peek wait: " << request_queue_->num_peek_waits();
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

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, alignment_result_buffer_list));
    OP_REQUIRES(ctx, request_queue_->push(shared_ptr<ResourceContainer<ReadResource>>(reads_container, resource_releaser)),
                Internal("Unable to push item onto work queue. Is it already closed?"));

    tracepoint(bioflow, snap_align_kernel, kernel_start, reads_container);
    tracepoint(bioflow, result_ready_queue_start, bufferlist_resource_container);
  }

private:
  clock_t kernel_start;

  struct time_log {
    std::chrono::high_resolution_clock::time_point end_subchunk;
    std::chrono::high_resolution_clock::time_point start_subchunk;
    std::chrono::high_resolution_clock::time_point ready;
    std::chrono::high_resolution_clock::time_point getnext;
    std::chrono::high_resolution_clock::time_point dropifequal;
    std::chrono::high_resolution_clock::time_point peek;

    void print() {

      auto subchunktime = std::chrono::duration_cast<std::chrono::microseconds>(start_subchunk - end_subchunk);
      auto readytime = std::chrono::duration_cast<std::chrono::microseconds>(ready - end_subchunk);
      auto getnexttime = std::chrono::duration_cast<std::chrono::microseconds>(getnext - ready);
      auto dropifequaltime = std::chrono::duration_cast<std::chrono::microseconds>(dropifequal - getnext);
      auto peektime = std::chrono::duration_cast<std::chrono::microseconds>(peek - dropifequal);
      LOG(INFO) << "subchunk time: " << subchunktime.count() 
        << " ready time: " << readytime.count()
        << " getnext time: " << getnexttime.count()
        << " dropifequal time: " << dropifequaltime.count()
        << " peek time: " << peektime.count();
    }
  };

  inline void init_workers(OpKernelContext* ctx) {
    auto aligner_func = [this] () {
      std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
      int my_id = 0;
      {
        mutex_lock l(mu_);
        my_id = thread_id_++;
      }

      /*cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(0, &cpuset);
      int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
      if (rc != 0) {
        LOG(INFO) << "Error calling pthread_setaffinity_np: " << rc << ", to core:0 " 
          << " for thread id: " << my_id;
      } else
        LOG(INFO) << "set affinity to core 0";*/
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

      /*LOG(INFO) << "reservation: " << BaseAligner::getBigAllocatorReservation(index, true,
            options->maxHits, MAX_READ_LENGTH, index->getSeedLength(), options->numSeedsFromCommandLine, options->seedCoverage, options->maxSecondaryAlignmentsPerContig)
          + alignmentResultBufferSize;*/

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

      BufferPair* result_buf = nullptr;
      ReadResource* subchunk_resource = nullptr;
      Status io_chunk_status, subchunk_status;
      const uint32_t max_completed = trace_granularity_;
      //std::chrono::high_resolution_clock::time_point end_subchunk = std::chrono::high_resolution_clock::now();
      //std::chrono::high_resolution_clock::time_point start_subchunk = std::chrono::high_resolution_clock::now();
     
      time_log timeLog;
      uint64 total = 0;
      timeLog.end_subchunk = std::chrono::high_resolution_clock::now();
      std::chrono::high_resolution_clock::time_point end_time;

      while (run_) {
        // reads must be in this scope for the custom releaser to work!
        shared_ptr<ResourceContainer<ReadResource>> reads_container;
        if (!request_queue_->peek(reads_container)) {
          continue;
        }
        //LOG(INFO) << "starting new chunk";
        //timeLog.peek = std::chrono::high_resolution_clock::now();

        auto *reads = reads_container->get();

        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &result_buf);
        while (io_chunk_status.ok()) {
          
          //timeLog.start_subchunk = std::chrono::high_resolution_clock::now();
          //auto subchunk_time = std::chrono::duration_cast<std::chrono::microseconds>(timeLog.start_subchunk - timeLog.end_subchunk);
          //if (subchunk_time.count() >= 500)
            //LOG(INFO) << "subchunk > 500";
            //timeLog.print();
          //total += subchunk_time.count();

          result_builder.set_buffer_pair(result_buf);
          //LOG(INFO) << "starting new subchunk";
          for (subchunk_status = subchunk_resource->get_next_record(&bases, &bases_len, &qualities, &qualities_len); subchunk_status.ok();
               subchunk_status = subchunk_resource->get_next_record(&bases, &bases_len, &qualities, &qualities_len)) {
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
                result_builder.AppendAlignmentResult(primaryResult, "*", 4);
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

            auto s = snap_wrapper::adjustResults(&snap_read, primaryResult, first_is_primary, format,
                                                 options_->useM, lvc, genome_, cigarString, flag);

            if (!s.ok())
              LOG(ERROR) << "computeCigarFlags did not return OK!!!";

            //auto t1 = std::chrono::high_resolution_clock::now();
            result_builder.AppendAlignmentResult(primaryResult, cigarString, flag);
            //auto t2 = std::chrono::high_resolution_clock::now();
            //auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            //LOG(INFO) << "append time " << time.count();
          }
          //timeLog.end_subchunk = std::chrono::high_resolution_clock::now();

          if (!IsResourceExhausted(subchunk_status)) {
            LOG(ERROR) << "Subchunk iteration ended without resource exhaustion!";
            compute_status_ = subchunk_status;
            return;
          }

          //result_builder.WriteResult(result_buf);
          result_buf->set_ready();
          //timeLog.ready = std::chrono::high_resolution_clock::now();
          //if (oldcapacity != newcapacity)
            //LOG(INFO) << "buffer reallocated";

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &result_buf);
         // timeLog.getnext = std::chrono::high_resolution_clock::now();
        }

        request_queue_->drop_if_equal(reads_container);
        //timeLog.dropifequal = std::chrono::high_resolution_clock::now();
        if (!IsResourceExhausted(io_chunk_status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError for I/O Chunk! : " << io_chunk_status << "\n";
          compute_status_ = io_chunk_status;
          return;
        }
        end_time = std::chrono::high_resolution_clock::now();
      }

      std::chrono::duration<double> thread_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
      struct rusage usage;
      int ret = getrusage(RUSAGE_THREAD, &usage);

      double total_s = (double)total / 1000000.0f;
      LOG(INFO) << "Aligner thread total time is: " << thread_time.count() << " seconds";
      LOG(INFO) << "Total time spent not processing" << total << " us";
      LOG(INFO) << "Total time spent not processing" << total_s << " seconds";
      LOG(INFO) << "system time used: " << usage.ru_stime.tv_sec << "." << usage.ru_stime.tv_usec << endl;
      LOG(INFO) << "user time used: " << usage.ru_utime.tv_sec << "." << usage.ru_utime.tv_usec << endl;
      LOG(INFO) << "maj page faults: " << usage.ru_minflt << endl;
      LOG(INFO) << "min page faults: " << usage.ru_majflt << endl;
      LOG(INFO) << "vol con sw: " << usage.ru_nvcsw << endl;
      LOG(INFO) << "invol con sw: " << usage.ru_nivcsw << endl;
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
  uint32_t trace_granularity_;

  atomic<uint32_t> num_active_threads_;
  mutex mu_;
  int thread_id_ = 0;

  int num_threads_;

  unique_ptr<WorkQueue<shared_ptr<ResourceContainer<ReadResource>>>> request_queue_;

  Status compute_status_;
  TF_DISALLOW_COPY_AND_ASSIGN(SnapAlignAGDParallelOp);
};

  REGISTER_OP("SnapAlignAGDParallel")
  .Attr("num_threads: int")
  .Attr("trace_granularity: int = 500")
  .Attr("chunk_size: int")
  .Attr("subchunk_size: int")
  .Attr("work_queue_size: int = 10")
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
