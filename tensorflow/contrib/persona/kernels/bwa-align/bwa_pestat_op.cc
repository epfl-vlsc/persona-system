#include <sys/time.h>
#include <sys/resource.h>
#include <vector>
#include <thread>
#include <memory>
#include <chrono>
#include <locale>
#include <pthread.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "bwa_wrapper.h"
#include "bwa_reads.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "bwa/bwamem.h"

namespace tensorflow {
using namespace std;
using namespace errors;

  class BWAPairedEndStatOp : public OpKernel {
    public:
      explicit BWAPairedEndStatOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      }

      ~BWAPairedEndStatOp() override {
        core::ScopedUnref index_unref(index_resource_);
        core::ScopedUnref options_unref(options_resource_);
      }

      void Compute(OpKernelContext* ctx) override {

        if (index_resource_ == nullptr) {
          OP_REQUIRES_OK(ctx, InitHandles(ctx));
        }

        ResourceContainer<BWAReadResource> *reads_container;
        OP_REQUIRES_OK(ctx, GetInput(ctx, "read", &reads_container));

        core::ScopedUnref a(reads_container);
        auto bwareads = reads_container->get();

        LOG(INFO) << "waiting for ready";
        bwareads->wait_for_ready();
        LOG(INFO) << "got ready";

        std::vector<mem_alnreg_v>& regs = bwareads->get_regs();
        mem_pestat_t* pes = bwareads->get_pes();
        LOG(INFO) << "pestat op calculating over " << regs.size() << " regs.";
        // set the pestat
        mem_pestat(bwa_options_, bwa_index_->bns->l_pac, regs.size(), &regs[0], pes);

        // thats all folks
      }

    private:

      Status InitHandles(OpKernelContext* ctx)
      {
        TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "options_handle", &options_resource_));
        TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "index_handle", &index_resource_));

        bwa_options_ = options_resource_->get();
        bwa_index_ = index_resource_->get();

        return Status::OK();
      }

      Status GetInput(OpKernelContext *ctx, const string &input_name, ResourceContainer<BWAReadResource> **reads_container)
      {
        const Tensor *read_input;
        TF_RETURN_IF_ERROR(ctx->input(input_name, &read_input));
        auto data = read_input->vec<string>(); // data(0) = container, data(1) = name
        TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(data(0), data(1), reads_container));

        LOG(INFO) << "stat setting output";
        ctx->set_output("read_handle", *read_input);
        /*Tensor* output;
        TF_RETURN_IF_ERROR(ctx->allocate_output(0, read_input->shape(), &output));
        auto out = output->vec<string>();
        out(0) = data(0);
        out(1) = data(1);*/
        // input is output as well
        return Status::OK();
      }

      BasicContainer<bwaidx_t> *index_resource_ = nullptr;
      BasicContainer<mem_opt_t>* options_resource_ = nullptr;
      bwaidx_t* bwa_index_ = nullptr;
      mem_opt_t *bwa_options_ = nullptr;
      TF_DISALLOW_COPY_AND_ASSIGN(BWAPairedEndStatOp);
  };

  REGISTER_KERNEL_BUILDER(Name("BWAPairedEndStat").Device(DEVICE_CPU), BWAPairedEndStatOp);

}  // namespace tensorflow
