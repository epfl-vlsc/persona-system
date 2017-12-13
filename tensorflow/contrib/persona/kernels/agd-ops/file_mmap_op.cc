#include <sys/types.h>
#include "tensorflow/contrib/persona/kernels/agd-format/shared_mmap_file_resource.h"
#include "tensorflow/contrib/persona/kernels/agd-format/memory_region.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  class FileMMapOp : public OpKernel {
  public:
    FileMMapOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("synchronous", &synchronous_));
    };

    ~FileMMapOp() {
      core::ScopedUnref unref_pool(ref_pool);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!ref_pool) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "pool_handle", &ref_pool));
      }
      const Tensor *filename_input;
      OP_REQUIRES_OK(ctx, ctx->input("filename", &filename_input));

      auto filename = filename_input->scalar<string>()();

      LOG(INFO) << "mapping file: " << filename;
      ResourceContainer<MemoryMappedFile> *mmf;
      OP_REQUIRES_OK(ctx, ref_pool->GetResource(&mmf));

      unique_ptr<ReadOnlyMemoryRegion> rmr;
      FileSystem *fs;
      OP_REQUIRES_OK(ctx, ctx->env()->GetFileSystemForFile(filename, &fs));
      OP_REQUIRES_OK(ctx, PosixMappedRegion::fromFile(filename, *fs, rmr, synchronous_));
      mmf->get()->own(move(rmr));

      OP_REQUIRES_OK(ctx, mmf->allocate_output("file_handle", ctx));
    }
  private:
    ReferencePool<MemoryMappedFile> *ref_pool = nullptr;
    bool synchronous_;
  };

  REGISTER_KERNEL_BUILDER(Name("FileMMap").Device(DEVICE_CPU), FileMMapOp);
} // namespace tensorflow {
