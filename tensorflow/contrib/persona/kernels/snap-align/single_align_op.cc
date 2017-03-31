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
#include "tensorflow/core/framework/queue_interface.h"
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

class SnapAlignSingleOp : public OpKernel {
  public:
    explicit SnapAlignSingleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_secondary", &max_secondary_));
      resource_container_shape_ = TensorShape({max_secondary_+1, 2});

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
      core::ScopedUnref queue_unref(queue_);
      while (num_active_threads_.load(memory_order_relaxed) > 0) {
        this_thread::sleep_for(chrono::milliseconds(10));
      }
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

    ResourceContainer<ReadResource> *reads_container;
    OP_REQUIRES_OK(ctx, GetInput(ctx, &reads_container));
    auto reads = reads_container->get();

    vector <ResourceContainer<BufferList>*> result_buffers(max_secondary_+1);
    OP_REQUIRES_OK(ctx, GetResultBufferLists(ctx, result_buffers));
    buffer_lists_.clear();
    for (auto rc : result_buffers) {
      buffer_lists_.push_back(rc->get());
    }

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, buffer_lists_));

  OP_REQUIRES(ctx, request_queue_->push(shared_ptr<ResourceContainer<ReadResource>>(reads_container, [this, ctx, result_buffers](ResourceContainer<ReadResource> *rr) {
      ResourceReleaser<ReadResource> a(*rr);
      {
        ReadResourceReleaser r(*rr->get());
        auto status = EnqueueBufferList(ctx, result_buffers);
        if (!status.ok()) {
          VLOG(ERROR) << "Enqueueing downstream buffer list failed for resoure container in SNAP Align Single";
          compute_status_ = status;
        }
      }
    })),
    Internal("Unable to push item onto work queue. Is it already closed?"));
  }


private:
  QueueInterface *queue_ = nullptr;
  ReferencePool<BufferList> *buflist_pool_ = nullptr;
  BasicContainer<GenomeIndex> *index_resource_ = nullptr;
  BasicContainer<AlignerOptions>* options_resource_ = nullptr;
  const Genome *genome_ = nullptr;
  AlignerOptions* options_ = nullptr;
  int subchunk_size_;
  volatile bool run_ = true;
  int max_secondary_;

  vector <BufferList*> buffer_lists_; // just used as a cache to proxy the ResourceContainer<BufferList> instances to split()

  atomic_uint_fast32_t num_active_threads_, id_{0};
  mutex mu_;

  int num_threads_;

  unique_ptr<ConcurrentQueue<shared_ptr<ResourceContainer<ReadResource>>>> request_queue_;

  Status compute_status_;
  TensorShape resource_container_shape_;

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

  Status InitHandles(OpKernelContext* ctx)
  {
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "options_handle", &options_resource_));
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "genome_handle", &index_resource_));
    TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
    TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 4), &queue_));
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

  Status GetInput(OpKernelContext* ctx, ResourceContainer<ReadResource> **reads_container) {
    const Tensor *input;
    TF_RETURN_IF_ERROR(ctx->input("read", &input));
    auto data = input->vec<string>();
    TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(data(0), data(1), reads_container));
    core::ScopedUnref a(*reads_container);

    return Status::OK();
  }

  Status GetResultBufferLists(OpKernelContext* ctx, vector<ResourceContainer<BufferList>*> &result_buffers) {
    ResourceContainer<BufferList> *ctr;
    result_buffers.clear();
    for (int i = 0; i < max_secondary_+1; i++) {
      TF_RETURN_IF_ERROR(buflist_pool_->GetResource(&ctr));
      ctr->get()->reset();
      result_buffers.push_back(ctr);
    }

    return Status::OK();
  }

  Status EnqueueBufferList(OpKernelContext *ctx, const vector<ResourceContainer<BufferList>*> &bl_ctr) {
    QueueInterface::Tuple tuple; // just a vector<Tensor>
    Tensor resource_container_out;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_STRING, resource_container_shape_, &resource_container_out));
    auto rc_out_matrix = resource_container_out.matrix<string>();
    auto max_dim = rc_out_matrix.dimension(0);
    if (bl_ctr.size() != max_dim) {
      return Internal("out matrix dim0 set at ", max_dim, " while vec<ResourceContainer> has size ", bl_ctr.size());
    }

    for (size_t i = 0; i < max_dim; i++) {
      rc_out_matrix(i, 0) = bl_ctr[i]->container();
      rc_out_matrix(i, 1) = bl_ctr[i]->name();
    }

    tuple.push_back(resource_container_out);

    TF_RETURN_IF_ERROR(queue_->ValidateTuple(tuple));

    // This is the synchronous version
    Notification n;
    queue_->TryEnqueue(tuple, ctx, [&n]() { n.Notify(); });
    n.WaitForNotification();

    return Status::OK();
  }

  inline void init_workers(OpKernelContext* ctx) {
    auto aligner_func = [this] () {
      std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
      int my_id = id_.fetch_add(1, memory_order_relaxed);

      int capacity = request_queue_->capacity();
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

      unique_ptr<BigAllocator> allocator(new BigAllocator(BaseAligner::getBigAllocatorReservation(index, true,
            options->maxHits, MAX_READ_LENGTH, index->getSeedLength(), options->numSeedsFromCommandLine, options->seedCoverage, options->maxSecondaryAlignmentsPerContig)
                                                          + alignmentResultBufferSize));

      /*LOG(INFO) << "reservation: " << BaseAligner::getBigAllocatorReservation(index, true,
            options->maxHits, MAX_READ_LENGTH, index->getSeedLength(), options->numSeedsFromCommandLine, options->seedCoverage, options->maxSecondaryAlignmentsPerContig)
          + alignmentResultBufferSize;*/

      BaseAligner* base_aligner = new (allocator.get()) BaseAligner(
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
        allocator.get()
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
      string cigarString;
      int flag;
      Read snap_read;
      LandauVishkinWithCigar lvc;

      vector<BufferPair*> result_bufs;
      ReadResource* subchunk_resource = nullptr;
      Status io_chunk_status, subchunk_status;

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

        auto *reads = reads_container->get();

        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        while (io_chunk_status.ok()) {

          if (result_bufs.size() > result_builders.size()) {
            result_builders.resize(result_bufs.size());
          }

          for (size_t i = 0; i < result_bufs.size(); i++) {
            result_builders[i].set_buffer_pair(result_bufs[i]);
          }

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

            // First, write the primary results
            auto s = snap_wrapper::WriteSingleResult(snap_read, primaryResult, result_builders[0], genome_, &lvc, false);

            if (!s.ok()) {
              LOG(ERROR) << "adjustResults did not return OK!!!";
              compute_status_ = s;
              return;
            }

            // Then write the secondary results if we specified them
            for (int i = 0; i < num_secondary_results; i++) {

              s = snap_wrapper::WriteSingleResult(snap_read, secondaryResults[i], result_builders[i+1], genome_, &lvc, true);
              if (!s.ok()) {
                LOG(ERROR) << "adjustResults did not return OK!!!";
                compute_status_ = s;
                return;
              }
            }
            for (int i = num_secondary_results; i < max_secondary_; i++) {
              // fill the columns with empties to maintain index equivalence
              result_builders[i].AppendEmpty();
            }
          }

          if (!IsResourceExhausted(subchunk_status)) {
            LOG(ERROR) << "Subchunk iteration ended without resource exhaustion!";
            compute_status_ = subchunk_status;
            return;
          }

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        }

        request_queue_->drop_if_equal(reads_container);
        if (!IsResourceExhausted(io_chunk_status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError for I/O Chunk! : " << io_chunk_status << "\n";
          compute_status_ = io_chunk_status;
          return;
        }
      }

      std::chrono::duration<double> thread_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
      /*struct rusage usage;
      int ret = getrusage(RUSAGE_THREAD, &usage);*/

      double total_s = (double)total / 1000000.0f;

      base_aligner->~BaseAligner(); // This calls the destructor without calling operator delete, allocator owns the memory.
      VLOG(INFO) << "base aligner thread ending.";
      num_active_threads_.fetch_sub(1, memory_order_relaxed);
    };
    auto worker_threadpool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    for (int i = 0; i < num_threads_; i++)
      worker_threadpool->Schedule(aligner_func);
    num_active_threads_ = num_threads_;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SnapAlignSingleOp);
};

  REGISTER_KERNEL_BUILDER(Name("SnapAlignSingle").Device(DEVICE_CPU), SnapAlignSingleOp);
}  // namespace tensorflow
