#include <vector>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

    namespace {
        const string op_name("StageBarrier");
    }

  class StageBarrierOp : public AsyncOpKernel {
  public:
    explicit StageBarrierOp(OpKernelConstruction *ctx) : AsyncOpKernel(ctx) { }

    ~StageBarrierOp() {
      core::ScopedUnref a1(input_queue_);
      core::ScopedUnref a2(output_queue_);
    }

    void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
      if (!input_queue_) {
        OP_REQUIRES_OK_ASYNC(ctx, Init(ctx), done);
      }

      const Tensor *input_name_t, *input_count_t;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("barrier_request_id", &input_name_t), done);
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input("barrier_request_count", &input_count_t), done);
      auto &input_name = input_name_t->scalar<string>()();
      auto &input_count = input_count_t->scalar<int32>()();

      expected_element_count_map_[input_name] = input_count;

      if (expected_element_count_map_.count(input_name) == 1) {
        vector<QueueInterface::Tuple> a;
        a.reserve(input_count);
        pending_element_map_.emplace(input_name, move(a));
      }

      //LOG(INFO) << "Barrier is waiting for input!!";
      while (!InputDone(input_name)) {
        OP_REQUIRES_OK_ASYNC(ctx, ProcessBarrierInput(ctx), done);
      }

      //LOG(INFO) << "barrier is dumping to downstream!";
      EnqueueAllDownstream(ctx, input_name);

      Tensor *output_name_t, *output_count_t;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output("request_id_out", TensorShape({}), &output_name_t), done);
      OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output("request_count_out", TensorShape({}), &output_count_t), done);

      output_name_t->scalar<string>()() = input_name;
      output_count_t->scalar<int32>()() = input_count;

      done();
    }
  private:
    QueueInterface *input_queue_ = nullptr, *output_queue_ = nullptr;
    unordered_map<string, vector<QueueInterface::Tuple>> pending_element_map_;
      unordered_map<string, size_t> expected_element_count_map_;

      inline Status
      ProcessBarrierInput(OpKernelContext *ctx) {
        Notification n;
          input_queue_->TryDequeue(ctx, [this, &n](const QueueInterface::Tuple &tuple) {
              auto &key = tuple[0];
              auto &key_name = key.scalar<string>()();
              pending_element_map_[key_name].push_back(tuple);
            n.Notify();
          });
        n.WaitForNotification();
        return Status::OK();
      }

      inline void
      EnqueueAllDownstream(OpKernelContext *ctx, const string &input_key) {
        auto &tuples = pending_element_map_.at(input_key);
        for (auto &tuple : tuples) {
          EnqueueTuple(ctx, tuple);
        }
        pending_element_map_.erase(input_key);
        expected_element_count_map_.erase(input_key);
      }

      inline void
      EnqueueTuple(OpKernelContext *ctx, const QueueInterface::Tuple &tuple) {
        Notification n;
        output_queue_->TryEnqueue(tuple, ctx, [&n]() {
            n.Notify();
        });
        n.WaitForNotification();
      }

      inline
      bool InputDone(const string &input_key) const {
        bool ret = (pending_element_map_.at(input_key).size() == expected_element_count_map_.at(input_key));
        return ret;
      }

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 2), &input_queue_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 3), &output_queue_));
      auto &input_dtypes = input_queue_->component_dtypes();
      auto &output_dtypes = output_queue_->component_dtypes();
      if (input_dtypes != output_dtypes) {
        return Internal("Barrier: Output and Input queues have different dtypes");
      }
      if (!(input_dtypes.at(0) == DT_STRING && output_dtypes.at(0) == DT_STRING)) {
        return Internal("Barrier: input or output queue has a non-string type for first element");
      }
      return Status::OK();
    }
  };

REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), StageBarrierOp);
} // namespace tensorflow {
