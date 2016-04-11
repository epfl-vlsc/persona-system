
// Stuart Byma
// Mostly copied from reader_op_kernel.cc


#include "tensorflow/core/framework/writer_op_kernel.h"

namespace tensorflow {

WriterOpKernel::WriterOpKernel(OpKernelConstruction* context)
    : OpKernel(context), have_handle_(false) {
  OP_REQUIRES_OK(context, context->allocate_persistent(
                              tensorflow::DT_STRING,
                              tensorflow::TensorShape({2}), &handle_, nullptr));
}

WriterOpKernel::~WriterOpKernel() {
  if (have_handle_ && cinfo_.resource_is_private_to_kernel()) {
    TF_CHECK_OK(cinfo_.resource_manager()->Delete<WriterInterface>(
        cinfo_.container(), cinfo_.name()));
  }
}

void WriterOpKernel::Compute(OpKernelContext* ctx) {
  mutex_lock l(mu_);
  if (!have_handle_) {
    OP_REQUIRES_OK(ctx, cinfo_.Init(ctx->resource_manager(), def(), false));
    WriterInterface* writer;
    OP_REQUIRES_OK(ctx,
                   cinfo_.resource_manager()->LookupOrCreate<WriterInterface>(
                       cinfo_.container(), cinfo_.name(), &writer,
                       [this](WriterInterface** ret) {
                         *ret = factory_();
                         return Status::OK();
                       }));
    writer->Unref();
    auto h = handle_.AccessTensor(ctx)->flat<string>();
    h(0) = cinfo_.container();
    h(1) = cinfo_.name();
    have_handle_ = true;
  }
  ctx->set_output_ref(0, &mu_, handle_.AccessTensor(ctx));
}

}  // namespace tensorflow
