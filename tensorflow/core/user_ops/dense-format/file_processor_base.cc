#include "file_processor_base.h"

namespace tensorflow {

  using namespace std;

  FileProcessorBase::FileProcessorBase(OpKernelConstruction *context) : OpKernel(context) {}

  void FileProcessorBase::Compute(OpKernelContext* ctx) {
    // TODO need to lock?
    // TODO this lookup may be incorrect
    const Tensor *resource_key;
    OP_REQUIRES_OK(ctx, ctx->input("file_handle", &resource_key));

    auto flat = resource_key->flat<string>();

    // try to get the file mapped resource
    MemoryMappedFile *mmf;
    OP_REQUIRES_OK(ctx,
                   GetResourceFromContext(ctx, flat(1), &mmf));
    core::ScopedUnref unref_me(mmf);
    // TODO scoped unref here
    // unref the resource of mmap file

    OP_REQUIRES_OK(ctx, ProcessFile(*mmf, ctx));

    /*
    // output the tensor of the new name
    ContainerInfo cinfo;
    OP_REQUIRES_OK(ctx, cinfo.Init(ctx->resource_manager(), def()));

    string name = "buff: " + flat(1);

    auto b = new BufferResource<char>(move(output_buf));
    OP_REQUIRES_OK(ctx, cinfo.resource_manager()->Create<BufferResource<char>>(
                                                                               cinfo.container(),
                                                                               name,
                                                                               b
                                                                               ));

    Tensor *output_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({2}), &output_tensor));
    auto flat_output = output_tensor->flat<string>();
    flat_output(0) = cinfo.container();
    flat_output(1) = name;
    */
  }
} // namespace tensorflow {
