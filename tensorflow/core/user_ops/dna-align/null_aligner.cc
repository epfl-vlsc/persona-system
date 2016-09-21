#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <vector>
#include <tuple>
#include <thread>
#include <memory>
#include <chrono>
#include <thread>
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

class NullAlignerOp : public OpKernel {
  public:
    explicit NullAlignerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("subchunk_size", &subchunk_size_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
      float wt;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("wait_time_secs", &wt));
      wait_time_ = wt * 1000000; // to put into microseconds
    }

    ~NullAlignerOp() override {
      core::ScopedUnref buflist_pool_unref(buflist_pool_);
    }

    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));

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
    //LOG(INFO) << "starting compute!";
    if (buflist_pool_ == nullptr) {
      OP_REQUIRES_OK(ctx, InitHandles(ctx));
    }

    auto start = chrono::high_resolution_clock::now();
    ResourceContainer<ReadResource> *reads_container;
    const Tensor *read_input;
    OP_REQUIRES_OK(ctx, ctx->input("read", &read_input));
    auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
    OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data(0), data(1), &reads_container));
    core::ScopedUnref a(reads_container);
    auto reads = reads_container->get();

    ResourceContainer<BufferList> *bufferlist_resource_container;
    OP_REQUIRES_OK(ctx, GetResultBufferList(ctx, &bufferlist_resource_container));
    auto alignment_result_buffer_list = bufferlist_resource_container->get();

    OP_REQUIRES_OK(ctx, reads->split(subchunk_size_, alignment_result_buffer_list));
    BufferPair* result_buf = nullptr;
    ReadResource* subchunk_resource = nullptr;
    const char *bases, *qualities;
    size_t bases_len, qualities_len;
    Status io_chunk_status, subchunk_status;
    io_chunk_status = Status::OK();
    io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &result_buf);
    while (io_chunk_status.ok()) {
        for (subchunk_status = subchunk_resource->get_next_record(&bases, &bases_len, &qualities, &qualities_len); subchunk_status.ok();
              subchunk_status = subchunk_resource->get_next_record(&bases, &bases_len, &qualities, &qualities_len)) {
          char size = static_cast<char>(bases_len);
          result_buf->index().AppendBuffer(&size, 1);
          result_buf->data().AppendBuffer(bases, 60);
        }

        result_buf->set_ready();
        io_chunk_status = reads->get_next_subchunk(&subchunk_resource, &result_buf);
    }
    resource_releaser(reads_container);
    auto end = chrono::high_resolution_clock::now();

    auto null_time = chrono::duration_cast<chrono::microseconds>(end - start);
    decltype(wait_time_) extra_wait = wait_time_ - null_time.count();
    if (extra_wait > 0) {
      //LOG(INFO) << "sleeping for " << extra_wait << " microseconds";
      usleep(extra_wait);
      //this_thread::sleep_for(chrono::microseconds(extra_wait));
    }
  }

private:


  ReferencePool<BufferList> *buflist_pool_ = nullptr;
  int subchunk_size_;
  int chunk_size_;
  int64_t wait_time_;


  Status compute_status_;
  TF_DISALLOW_COPY_AND_ASSIGN(NullAlignerOp);
};

  REGISTER_OP("NullAligner")
  .Attr("chunk_size: int")
  .Attr("subchunk_size: int")
  .Attr("wait_time_secs: float = 0.0")
  .Input("buffer_list_pool: Ref(string)")
  .Input("read: string")
  .Output("result_buf_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Aligns input `read`, which contains multiple reads.
wait_time specifies the minimum time that the alignment should take
Loads the SNAP-based hash table into memory on construction to perform
generation of alignment candidates.
outputs a tensor [num_reads] containing serialized reads and results
containing the alignment candidates.
)doc");

  REGISTER_KERNEL_BUILDER(Name("NullAligner").Device(DEVICE_CPU), NullAlignerOp);

}  // namespace tensorflow
