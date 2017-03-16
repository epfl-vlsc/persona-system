#include <sys/types.h>
#include <sys/stat.h>
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "tensorflow/core/platform/logging.h"
#include <string>

namespace tensorflow {

  using namespace std;
  using namespace errors;

  class StagedFileMapOp : public OpKernel {
  public:
    StagedFileMapOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("local_prefix", &path_prefix_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("synchronous", &synchronous_));

      struct stat info;
      OP_REQUIRES(ctx, stat(path_prefix_.c_str(), &info) == 0,
                  Internal("Unable to call stat on: ", path_prefix_));
      OP_REQUIRES(ctx, info.st_mode & S_IFDIR != 0,
                  Internal("Local prefix is not a valid directory: ", path_prefix_));
      auto a = path_prefix_.find_last_of("/");
      OP_REQUIRES(ctx, a != string::npos, Internal("Invalid local prefix: ", a));
      if (a < path_prefix_.length()-1) {
        path_prefix_.append("/");
      }
    }

    ~StagedFileMapOp() {
      core::ScopedUnref unref_pool(ref_pool);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!ref_pool) {
        OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "pool_handle", &ref_pool));
      }
      const Tensor *upstream_refs, *upstream_names, *filename_input;
      OP_REQUIRES_OK(ctx, ctx->input("upstream_names", &upstream_names));
      OP_REQUIRES_OK(ctx, ctx->input("upstream_refs", &upstream_refs));

      OP_REQUIRES_OK(ctx, ctx->input("filename", &filename_input));

      auto file_key = filename_input->scalar<string>()();
      tracepoint(bioflow, process_key, file_key.c_str());

      auto filename = path_prefix_ + file_key;

      start = chrono::high_resolution_clock::now();

      ResourceContainer<MemoryMappedFile> *mmf;
      OP_REQUIRES_OK(ctx, ref_pool->GetResource(&mmf));

      unique_ptr<ReadOnlyMemoryRegion> rmr;
      OP_REQUIRES_OK(ctx, ctx->env()->NewReadOnlyMemoryRegionFromFile(filename, &rmr, synchronous_));
      mmf->get()->own(move(rmr));

      Tensor *file_handles, *file_names;
      TensorShape file_handles_shape(upstream_refs->shape());
      TensorShape file_names_shape(upstream_names->shape());
      file_handles_shape.set_dim(0, file_handles_shape.dim_size(0) + 1);
      file_names_shape.set_dim(0, file_names_shape.dim_size(0) + 1);
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_handles", file_handles_shape, &file_handles));
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_names", file_names_shape, &file_names));

      auto handles_matrix = file_handles->matrix<string>();
      auto names_vec = file_names->vec<string>();
      auto upstream_handles_matrix = upstream_refs->matrix<string>();
      auto upstream_names_vec = upstream_names->vec<string>();

      auto max_dim = upstream_refs->dim_size(0);
      for (int i = 0; i < max_dim; i++) {
        names_vec(i) = upstream_names_vec(i);
        handles_matrix(i, 0) = upstream_handles_matrix(i, 0);
        handles_matrix(i, 1) = upstream_handles_matrix(i, 1);
      }

      names_vec(max_dim) = filename;
      handles_matrix(max_dim, 0) = mmf->container();
      handles_matrix(max_dim, 1) = mmf->name();

      auto duration = TRACEPOINT_DURATION_CALC(start);
      tracepoint(bioflow, chunk_read, file_key.c_str(), duration);
    }
  private:
    chrono::high_resolution_clock::time_point start;
    ReferencePool<MemoryMappedFile> *ref_pool = nullptr;
    string path_prefix_;
    bool synchronous_;
  };

  class FileMMapOp : public OpKernel {
  public:
    FileMMapOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("local_prefix", &path_prefix_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("synchronous", &synchronous_));

      struct stat info;
      OP_REQUIRES(ctx, stat(path_prefix_.c_str(), &info) == 0,
                  Internal("Unable to call stat on: ", path_prefix_));
      OP_REQUIRES(ctx, info.st_mode & S_IFDIR != 0,
                  Internal("Local prefix is not a valid directory: ", path_prefix_));
      auto a = path_prefix_.find_last_of("/");
      OP_REQUIRES(ctx, a != string::npos, Internal("Invalid local prefix: ", a));
      if (a < path_prefix_.length()-1) {
        path_prefix_.append("/");
      }
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

      auto file_key = filename_input->scalar<string>()();
      tracepoint(bioflow, process_key, file_key.c_str());
      auto filename = path_prefix_ + file_key;

      start = chrono::high_resolution_clock::now();

      ResourceContainer<MemoryMappedFile> *mmf;
      OP_REQUIRES_OK(ctx, ref_pool->GetResource(&mmf));

      unique_ptr<ReadOnlyMemoryRegion> rmr;
      OP_REQUIRES_OK(ctx, ctx->env()->NewReadOnlyMemoryRegionFromFile(filename, &rmr)); //, synchronous_));
      mmf->get()->own(move(rmr));

      Tensor *output_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_handle", TensorShape({2}), &output_tensor));
      auto output_vector = output_tensor->vec<string>();
      output_vector(0) = mmf->container();
      output_vector(1) = mmf->name();

      Tensor *file_name;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_name", TensorShape({}), &file_name));
      auto scalar = file_name->scalar<string>();
      scalar() = filename;

      auto duration = TRACEPOINT_DURATION_CALC(start);
      tracepoint(bioflow, chunk_read, filename.c_str(), duration);
    }
  private:
    chrono::high_resolution_clock::time_point start;
    ReferencePool<MemoryMappedFile> *ref_pool = nullptr;
    string path_prefix_;
    bool synchronous_;
  };

  REGISTER_KERNEL_BUILDER(Name("FileMMap").Device(DEVICE_CPU), FileMMapOp);
  REGISTER_KERNEL_BUILDER(Name("StagedFileMap").Device(DEVICE_CPU), StagedFileMapOp);

} // namespace tensorflow {
