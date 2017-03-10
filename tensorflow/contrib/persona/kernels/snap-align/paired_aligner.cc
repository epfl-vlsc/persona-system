#include <sys/time.h>
#include <sys/resource.h>
#include <array>
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
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer_list.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/FileFormat.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/ChimericPairedEndAligner.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/IntersectingPairedEndAligner.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/PairedAligner.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/AlignmentResult.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
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

class AGDPairedAlignerOp : public OpKernel {
  public:
    explicit AGDPairedAlignerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("sam_format", &sam_format_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("max_secondary", &max_secondary_));
      subchunk_size_ *= 2;

      int capacity;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("work_queue_size", &capacity));
      request_queue_.reset(new ConcurrentQueue<shared_ptr<ResourceContainer<ReadResource>>>(capacity));
      compute_status_ = Status::OK();
    }

    ~AGDPairedAlignerOp() override {
      if (!run_) {
        LOG(ERROR) << "Unable to safely wait in ~AGDPairedAlignerOp for all threads. run_ was toggled to false\n";
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
      VLOG(DEBUG) << "AGD Align Destructor(" << this << ") finished\n";
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
    OP_REQUIRES_OK(ctx, GetInput(ctx, "read", &reads_container));

    core::ScopedUnref a(reads_container);
    auto reads = reads_container->get();

    OP_REQUIRES_OK(ctx, GetResultBufferLists(ctx));

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, buffer_lists_));

    if (sam_format_) {
      OP_REQUIRES(ctx, request_queue_->push(shared_ptr<ResourceContainer<ReadResource>>(reads_container, no_resource_releaser)), Internal("Unable to push item onto work queue. Is it already closed?"));
    } else {
      OP_REQUIRES(ctx, request_queue_->push(shared_ptr<ResourceContainer<ReadResource>>(reads_container, resource_releaser)), Internal("Unable to push item onto work queue. Is it already closed?"));
    }
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

  Status GetInput(OpKernelContext *ctx, const string &input_name, ResourceContainer<ReadResource> **reads_container)
  {
    const Tensor *read_input;
    TF_RETURN_IF_ERROR(ctx->input(input_name, &read_input));
    auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
    TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(data(0), data(1), reads_container));
    return Status::OK();
  }

  Status GetResultBufferLists(OpKernelContext* ctx)
  {
    ResourceContainer<BufferList> **ctr;
    Tensor* out_t;
    buffer_lists_.clear();
    buffer_lists_.reserve(max_secondary_+1);
    TF_RETURN_IF_ERROR(ctx->allocate_output("result_buf_handle", TensorShape({max_secondary_+1, 2}), &out_t));
    auto out_matrix = out_t->matrix<string>();
    for (int i = 0; i < max_secondary_+1; i++) {
      TF_RETURN_IF_ERROR(buflist_pool_->GetResource(ctr));
      //core::ScopedUnref a(reads_container);
      (*ctr)->get()->reset();
      buffer_lists_.push_back((*ctr)->get());
      out_matrix(i, 0) = (*ctr)->container();
      out_matrix(i, 1) = (*ctr)->name();
    }

    return Status::OK();
  }

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

      snap_wrapper::PairedAligner aligner(options_, index, max_secondary_);

      PairedAlignmentResult* secondary_results = new PairedAlignmentResult[aligner.MaxPairedSecondary()];
      SingleAlignmentResult* secondary_single_results = new SingleAlignmentResult[aligner.MaxSingleSecondary()];
      bool first_is_primary = true; // we only ever generate one result

      PairedAlignmentResult primaryResult;
      vector<AlignmentResultBuilder> result_builders;
      string cigarString;
      int flag;
      //Read snap_read[2];
      array<Read, 2> snap_read;

      LandauVishkinWithCigar lvc;

      vector<BufferPair*> result_bufs;
      ReadResource* subchunk_resource = nullptr;
      Status io_chunk_status, subchunk_status;
      //std::chrono::high_resolution_clock::time_point end_subchunk = std::chrono::high_resolution_clock::now();
      //std::chrono::high_resolution_clock::time_point start_subchunk = std::chrono::high_resolution_clock::now();
      bool useless[2], pass[2], pass_all;
      int num_secondary_results, num_secondary_single_results_first, 
          num_secondary_single_results_second;

      while (run_) {
        // reads must be in this scope for the custom releaser to work!
        shared_ptr<ResourceContainer<ReadResource>> reads_container;
        if (!request_queue_->peek(reads_container)) {
          continue;
        }
        //timeLog.peek = std::chrono::high_resolution_clock::now();

        auto *reads = reads_container->get();

        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        while (io_chunk_status.ok()) {

          for (int i = 0; i < result_builders.size(); i++)
            result_builders[i].set_buffer_pair(result_bufs[i]);

          subchunk_status = Status::OK();
          while (subchunk_status.ok()) {
            for (size_t i = 0; i < 2; ++i) {
              auto &sread = snap_read[i];
              subchunk_status = subchunk_resource->get_next_record(sread);
              if (subchunk_status.ok()) {
                sread.clip(options_->clipping);
                useless[i] = sread.getDataLength() < options_->minReadLength && sread.countOfNs() > options_->maxDist;
              } else {
                break;
              }
            }

            if (!subchunk_status.ok()) {
              break;
            }

            if (useless[0] || useless[1]) {
                // TODO change to the writePairs code for the new readWriter
              // we just filter these out for now
              for (size_t i = 0; i < 2; ++i) {
                primaryResult.status[i] = AlignmentResult::NotFound;
                primaryResult.location[i] = InvalidGenomeLocation;
                primaryResult.mapq[i] = 0;
                primaryResult.direction[i] = FORWARD; 
                subchunk_status = aligner.writeResult(snap_read, primaryResult, result_builders[0], false);
                // fill in blanks for secondaries
                for (int i = 1; i < result_builders.size(); i++) {
                  result_builders[i].AppendEmpty();
                }
              }
              continue;
            }

            aligner.align(snap_read, primaryResult, secondary_results, &num_secondary_results,
                secondary_single_results, &num_secondary_single_results_first, &num_secondary_single_results_second);
            subchunk_status = aligner.writeResult(snap_read, primaryResult, result_builders[0], false);
            int i = 0;
            // we either have paired secondaries, or single ended results for each, but not both
            if (num_secondary_results > 0) {
              while (subchunk_status.ok() && i < num_secondary_results) {
                subchunk_status = aligner.writeResult(snap_read, secondary_results[i], result_builders[i], true);
                i++;
              }
            } else if (num_secondary_single_results_first > 0 || num_secondary_single_results_second > 0) {
              while (subchunk_status.ok() && i < num_secondary_single_results_first) {
                subchunk_status = snap_wrapper::WriteSingleResult(snap_read[0], secondary_single_results[i],
                    result_builders[i+1], index->getGenome(), &lvc, true);
                i++;
              }
              i = 0;
              while (subchunk_status.ok() && i < num_secondary_single_results_second) {
                subchunk_status = snap_wrapper::WriteSingleResult(snap_read[1], secondary_single_results[i+num_secondary_single_results_first],
                    result_builders[i+1], index->getGenome(), &lvc, true);
                i++;
              }
              // fill in the gaps
              i = 0;
              while (num_secondary_single_results_first + i < num_secondary_single_results_second) {
                result_builders[num_secondary_single_results_first + i + 1].AppendEmpty();
                i++;
              }
              i = 0;
              while (num_secondary_single_results_second + i < num_secondary_single_results_first) {
                result_builders[num_secondary_single_results_first + i + 1].AppendEmpty();
                i++;
              }
            }
          }

          if (!IsResourceExhausted(subchunk_status)) {
            LOG(ERROR) << "Subchunk iteration ended without resource exhaustion!";
            compute_status_ = subchunk_status;
            return;
          }

          for (auto buf : result_bufs)
            buf->set_ready();

          io_chunk_status = reads->get_next_subchunk(&subchunk_resource, result_bufs);
        }

        request_queue_->drop_if_equal(reads_container);

        if (!IsResourceExhausted(io_chunk_status)) {
          LOG(ERROR) << "Aligner thread received non-ResourceExhaustedError for I/O Chunk! : " << io_chunk_status << "\n";
          compute_status_ = io_chunk_status;
          return;
        }
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
  BasicContainer<GenomeIndex> *index_resource_ = nullptr;
  BasicContainer<PairedAlignerOptions>* options_resource_ = nullptr;
  const Genome *genome_ = nullptr;
  PairedAlignerOptions *options_ = nullptr;
  int subchunk_size_;
  bool sam_format_;
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
  TF_DISALLOW_COPY_AND_ASSIGN(AGDPairedAlignerOp);
};

  REGISTER_KERNEL_BUILDER(Name("AGDPairedAligner").Device(DEVICE_CPU), AGDPairedAlignerOp);

}  // namespace tensorflow
