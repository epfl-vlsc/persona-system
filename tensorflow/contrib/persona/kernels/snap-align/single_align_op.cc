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
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/FileFormat.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "GenomeIndex.h"
#include "Read.h"
#include "SnapAlignerWrapper.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"

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

    void no_resource_releaser(ResourceContainer<ReadResource> *rr) {
      // nothing to do
    }
  }

class SnapAlignSingleOp : public OpKernel {
  public:
    explicit SnapAlignSingleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_secondary", &max_secondary_));

      int capacity;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("work_queue_size", &capacity));
      request_queue_.reset(new ConcurrentQueue<shared_ptr<ResourceContainer<ReadResource>>>(capacity));
      compute_status_ = Status::OK();
    }

    ~SnapAlignSingleOp() override {
      if (!run_) {
        LOG(ERROR) << "Unable to safely wait in ~SnapAlignSingleOp for all threads. run_ was toggled to false\n";
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
      auto avg_inter_time = total_usec / total_invoke_intervals;
      LOG(INFO) << "average inter kernel time: " << avg_inter_time;
      //LOG(INFO) << "done queue push wait: " << done_queue_->num_push_waits();
      //LOG(INFO) << "done queue pop wait: " << done_queue_->num_pop_waits();
      VLOG(INFO) << "AGD Align Destructor(" << this << ") finished\n";
    }

    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "options_handle", &options_resource_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "genome_handle", &index_resource_));
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
      TF_RETURN_IF_ERROR(snap_wrapper::init());

      options_ = options_resource_->get();
      genome_ = index_resource_->get()->getGenome();

      /*if (options_->maxSecondaryAlignmentAdditionalEditDistance < 0) {
        num_secondary_alignments_ = 0;
      } else {
        num_secondary_alignments_ = BaseAligner::getMaxSecondaryResults(options_->numSeedsFromCommandLine,
            options_->seedCoverage, MAX_READ_LENGTH, options_->maxHits, index_resource_->get_index()->getSeedLength());
      }*/

      return Status::OK();
    }

    Status GetResultBufferLists(OpKernelContext* ctx)
    {
      ResourceContainer<BufferList> *ctr;
      Tensor* out_t;
      buffer_lists_.clear();
      buffer_lists_.reserve(max_secondary_+1);
      TF_RETURN_IF_ERROR(ctx->allocate_output("result_buf_handle", TensorShape({max_secondary_+1, 2}), &out_t));
      auto out_matrix = out_t->matrix<string>();
      for (int i = 0; i < max_secondary_+1; i++) {
        TF_RETURN_IF_ERROR(buflist_pool_->GetResource(&ctr));
        //core::ScopedUnref a(reads_container);
        ctr->get()->reset();
        buffer_lists_.push_back(ctr->get());
        out_matrix(i, 0) = ctr->container();
        out_matrix(i, 1) = ctr->name();
      }

      return Status::OK();
    }

  void Compute(OpKernelContext* ctx) override {
    if (first) {
      first = false;
    } else {
      t_now = std::chrono::high_resolution_clock::now();
      auto interval_time = std::chrono::duration_cast<std::chrono::microseconds>(t_now - t_last);
      total_usec += interval_time.count();
      total_invoke_intervals++;
    }
      

    if (index_resource_ == nullptr) {
      OP_REQUIRES_OK(ctx, InitHandles(ctx));
      init_workers(ctx);
    }

    if (!compute_status_.ok()) {
      ctx->SetStatus(compute_status_);
      return;
    }

    OP_REQUIRES(ctx, run_, Internal("One of the aligner threads triggered a shutdown of the aligners. Please inspect!"));

    ResourceContainer<ReadResource> *reads_container;
    const Tensor *read_input;
    OP_REQUIRES_OK(ctx, ctx->input("read", &read_input));
    auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &reads_container));
    core::ScopedUnref a(reads_container);
    auto reads = reads_container->get();

    OP_REQUIRES_OK(ctx, GetResultBufferLists(ctx));

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, buffer_lists_));

    OP_REQUIRES(ctx, request_queue_->push(shared_ptr<ResourceContainer<ReadResource>>(reads_container, resource_releaser)),
              Internal("Unable to push item onto work queue. Is it already closed?"));
    t_last = std::chrono::high_resolution_clock::now();
  }


private:
  uint64 total_usec = 0;
  uint64 total_invoke_intervals = 0;
  bool first = true;
  std::chrono::high_resolution_clock::time_point t_now;
  std::chrono::high_resolution_clock::time_point t_last;

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
      auto index = index_resource_->get();
      auto options = options_resource_->get();

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

      const char *bases, *qualities;
      size_t bases_len, qualities_len;
      SingleAlignmentResult primaryResult;
      vector<SingleAlignmentResult> secondaryResults;
      secondaryResults.resize(max_secondary_);

      int num_secondary_results;
      SAMFormat format(options_->useM);
      vector<AlignmentResultBuilder> result_builders;
      result_builders.resize(1+max_secondary_);
      string cigarString;
      int flag;
      Read snap_read;
      LandauVishkinWithCigar lvc;

      vector<BufferPair*> result_bufs;
      ReadResource* subchunk_resource = nullptr;
      Status io_chunk_status, subchunk_status;
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

        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        while (io_chunk_status.ok()) {

          //timeLog.start_subchunk = std::chrono::high_resolution_clock::now();
          //auto subchunk_time = std::chrono::duration_cast<std::chrono::microseconds>(timeLog.start_subchunk - timeLog.end_subchunk);
          //if (subchunk_time.count() >= 500)
            //LOG(INFO) << "subchunk > 500";
            //timeLog.print();
          //total += subchunk_time.count();

          for (int i = 0; i < result_builders.size(); i++)
            result_builders[i].set_buffer_pair(result_bufs[i]);
          //LOG(INFO) << "starting new subchunk";
          for (subchunk_status = subchunk_resource->get_next_record(snap_read); subchunk_status.ok();
               subchunk_status = subchunk_resource->get_next_record(snap_read)) {
            cigarString.clear();
            snap_read.clip(options_->clipping);
            if (snap_read.getDataLength() < options_->minReadLength || snap_read.countOfNs() > options_->maxDist) {
              primaryResult.status = AlignmentResult::NotFound;
              primaryResult.location = InvalidGenomeLocation;
              primaryResult.mapq = 0;
              primaryResult.direction = FORWARD;
              //result_builder.AppendAlignmentResult(primaryResult, "*", 4);
              auto s = snap_wrapper::WriteSingleResult(snap_read, primaryResult, result_builders[0], genome_, &lvc, false);

              if (!s.ok()) {
                LOG(ERROR) << "adjustResults did not return OK!!!";
              }
              for (int i = 1; i < max_secondary_; i++) {
                // fill the columns with empties to maintain index equivalence
                result_builders[i].AppendEmpty();
              }
              continue;
            }

            base_aligner->AlignRead(
                                    &snap_read,
                                    &primaryResult,
                                    options_->maxSecondaryAlignmentAdditionalEditDistance,
                                    alignmentResultBufferSize,
                                    &num_secondary_results,
                                    max_secondary_,
                                    &secondaryResults[0] //secondaryResults
                                    );

            flag = 0;

            auto s = snap_wrapper::WriteSingleResult(snap_read, primaryResult, result_builders[0], genome_, &lvc, false);

            if (!s.ok()) {
              LOG(ERROR) << "adjustResults did not return OK!!!";
            }
            
            for (int i = 0; i < num_secondary_results; i++) {
            
              auto s = snap_wrapper::WriteSingleResult(snap_read, secondaryResults[i], result_builders[i+1], genome_, &lvc, true);
              if (!s.ok()) {
                LOG(ERROR) << "adjustResults did not return OK!!!";
              }
            }
            for (int i = num_secondary_results; i < max_secondary_; i++) {
              // fill the columns with empties to maintain index equivalence
              result_builders[i].AppendEmpty();
            }


          }
          //timeLog.end_subchunk = std::chrono::high_resolution_clock::now();

          if (!IsResourceExhausted(subchunk_status)) {
            LOG(ERROR) << "Subchunk iteration ended without resource exhaustion!";
            compute_status_ = subchunk_status;
            return;
          }

          for (auto buf : result_bufs)
            buf->set_ready();
          //timeLog.ready = std::chrono::high_resolution_clock::now();
          //if (oldcapacity != newcapacity)
            //LOG(INFO) << "buffer reallocated";

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
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
      /*struct rusage usage;
      int ret = getrusage(RUSAGE_THREAD, &usage);*/

      double total_s = (double)total / 1000000.0f;
      /*LOG(INFO) << "Aligner thread total time is: " << thread_time.count() << " seconds";
      LOG(INFO) << "Total time spent not processing" << total << " us";
      LOG(INFO) << "Total time spent not processing" << total_s << " seconds";
      LOG(INFO) << "system time used: " << usage.ru_stime.tv_sec << "." << usage.ru_stime.tv_usec << endl;
      LOG(INFO) << "user time used: " << usage.ru_utime.tv_sec << "." << usage.ru_utime.tv_usec << endl;
      LOG(INFO) << "maj page faults: " << usage.ru_minflt << endl;
      LOG(INFO) << "min page faults: " << usage.ru_majflt << endl;
      LOG(INFO) << "vol con sw: " << usage.ru_nvcsw << endl;
      LOG(INFO) << "invol con sw: " << usage.ru_nivcsw << endl;*/
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
  BasicContainer<GenomeIndex> *index_resource_ = nullptr;
  BasicContainer<AlignerOptions>* options_resource_ = nullptr;
  const Genome *genome_ = nullptr;
  AlignerOptions* options_ = nullptr;
  int subchunk_size_;
  volatile bool run_ = true;
  uint64_t id_ = 0;
  int max_secondary_;
  vector<BufferList*> buffer_lists_;

  atomic<uint32_t> num_active_threads_;
  mutex mu_;
  int thread_id_ = 0;

  int num_threads_;

  unique_ptr<ConcurrentQueue<shared_ptr<ResourceContainer<ReadResource>>>> request_queue_;

  Status compute_status_;
  TF_DISALLOW_COPY_AND_ASSIGN(SnapAlignSingleOp);
};

  REGISTER_KERNEL_BUILDER(Name("SnapAlignSingle").Device(DEVICE_CPU), SnapAlignSingleOp);

}  // namespace tensorflow
