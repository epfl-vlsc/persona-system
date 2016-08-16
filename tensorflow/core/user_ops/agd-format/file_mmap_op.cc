#include <sys/types.h>
#include <sys/stat.h>
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"
#include "tensorflow/core/platform/logging.h"
#include <string>

namespace tensorflow {

  using namespace std;
  using namespace errors;

  REGISTER_OP("FileMMap")
  .Attr("local_prefix: string = ''")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("pool_handle: Ref(string)")
  .Input("filename: string")
  .Output("file_handle: string")
  .Output("file_name: string")
  .SetIsStateful()
  .Doc(R"doc(
Produces memory-mapped files, synchronously reads them, and produces a Tensor<2>
with the container and shared name for the file.

local_prefix: a directory on the local machine on which to find the keys
This is used in the case of a remote reader giving only the filenames to this reader
queue_handle: a handle to the filename queue
file_handle: a Tensor(2) of strings to access the shared mmaped file resource to downstream nodes
file_name: a Tensor() of string for the unique key for this file
  )doc");

  REGISTER_OP("StagedFileMap")
  .Attr("local_prefix: string = ''")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Input("filename: string")
  .Input("upstream_refs: string")
  .Input("upstream_names: string")
  .Input("pool_handle: Ref(string)")
  .Output("file_handles: string")
  .Output("file_names: string")
  .SetIsStateful()
  .Doc(R"doc(
Appends a agd reader handle tensor to an input list.
To be used for the staged pipeline

local_prefix: a directory on the local machine on which to find the keys
This is used in the case of a remote reader giving only the filenames to this reader
queue_handle: handle to the filename queue
upstream: the handles from previous stages in the pipeline, if any
upstream_name: the names from the previous stages of the pipeline, if any
bundle: [{this file map op}] + upstream
bundle_name: [{this map op's name}] + upstream_name
)doc");

  class StagedFileMapOp : public OpKernel {
  public:
    StagedFileMapOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("local_prefix", &path_prefix_));

      struct stat info;
      OP_REQUIRES(ctx, stat(path_prefix_.c_str(), &info) == 0,
                  Internal("Unable to call stat on: ", path_prefix_));
      OP_REQUIRES(ctx, info.st_mode & S_IFDIR != 0,
                  Internal("Local prefix is not a valid directory: ", path_prefix_));
    }

    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      start = clock();
      const Tensor *upstream_refs, *upstream_names, *filename_input;
      OP_REQUIRES_OK(ctx, ctx->input("upstream_names", &upstream_names));
      OP_REQUIRES_OK(ctx, ctx->input("upstream_refs", &upstream_refs));

      OP_REQUIRES_OK(ctx, ctx->input("filename", &filename_input));

      auto filename = path_prefix_ + filename_input->scalar<string>()();

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));

      ReferencePool<MemoryMappedFile> *ref_pool;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "pool_handle", &ref_pool));
      core::ScopedUnref unref_pool(ref_pool);

      ResourceContainer<MemoryMappedFile> *mmf;
      OP_REQUIRES_OK(ctx, ref_pool->GetResource(&mmf));

      auto start = clock();
      ReadOnlyMemoryRegion *rmr;
      OP_REQUIRES_OK(ctx, ctx->env()->NewReadOnlyMemoryRegionFromFile(filename, &rmr));
      mmf->get()->own(rmr);

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
      tracepoint(bioflow, read_kernel, start, filename.c_str(), mmf->get()->size());
      tracepoint(bioflow, read_ready_queue_start, mmf);
    }
  private:
    clock_t start;
    string path_prefix_;
  };

  class FileMMapOp : public OpKernel {
  public:
    FileMMapOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("local_prefix", &path_prefix_));

      struct stat info;
      OP_REQUIRES(ctx, stat(path_prefix_.c_str(), &info) == 0,
                  Internal("Unable to call stat on: ", path_prefix_));
      OP_REQUIRES(ctx, info.st_mode & S_IFDIR != 0,
                  Internal("Local prefix is not a valid directory: ", path_prefix_));
    };

    void Compute(OpKernelContext* ctx) override {
      const Tensor *filename_input;
      OP_REQUIRES_OK(ctx, ctx->input("filename", &filename_input));
      auto filename = path_prefix_ + filename_input->scalar<string>()();

      ContainerInfo cinfo;
      OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));

      ReferencePool<MemoryMappedFile> *ref_pool;
      OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "pool_handle", &ref_pool));
      core::ScopedUnref unref_pool(ref_pool);

      ResourceContainer<MemoryMappedFile> *mmf;
      OP_REQUIRES_OK(ctx, ref_pool->GetResource(&mmf));

      ReadOnlyMemoryRegion *rmr;
      OP_REQUIRES_OK(ctx, ctx->env()->NewReadOnlyMemoryRegionFromFile(filename, &rmr));
      mmf->get()->own(rmr);

      Tensor *output_tensor;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_handle", TensorShape({1, 2}), &output_tensor));
      auto output_matrix = output_tensor->matrix<string>();
      output_matrix(0, 0) = mmf->container();
      output_matrix(0, 1) = mmf->name();

      Tensor *file_name;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("file_name", TensorShape({1}), &file_name));
      auto scalar = file_name->vec<string>();
      scalar(0) = filename;
      tracepoint(bioflow, read_kernel, start, filename.c_str(), mmf->get()->size());
      tracepoint(bioflow, read_ready_queue_start, mmf);
    }
  private:
    clock_t start;
    string path_prefix_;
  };

  REGISTER_KERNEL_BUILDER(Name("FileMMap").Device(DEVICE_CPU), FileMMapOp);
  REGISTER_KERNEL_BUILDER(Name("StagedFileMap").Device(DEVICE_CPU), StagedFileMapOp);

} // namespace tensorflow {
