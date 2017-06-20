#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/queue_interface.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  class BatcherOp : public OpKernel {
  public:
    BatcherOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &expected_shape_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    }

    ~BatcherOp() {
      core::ScopedUnref a1(input_queue_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!input_queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }
      current_record_id_.clear();
      bool emitted = false;
      while (!emitted && batch_.size() < batch_size_) {
        OP_REQUIRES_OK(ctx, DequeueElement(ctx, emitted));
      }
      if (!emitted) {
        OP_REQUIRES_OK(ctx, EnqueueOutput(ctx));
      }
    }

  private:
    TensorShape expected_shape_;
    DataType dtype_;
    int batch_size_;
    string current_record_id_;
    QueueInterface *input_queue_ = nullptr;
    vector<Tensor> batch_;

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &input_queue_));
      auto &dtypes = input_queue_->component_dtypes();
      if (dtypes.size() != 2) {
        return Internal("queue must have 2 elements, the first being the string id and latter being the desired id");
      }
      auto &record_id_type = dtypes[0];
      auto &value_type = dtypes[1];
      if (record_id_type != DT_STRING) {
        return Internal("first element of queue type must be string for record id");
      }

      if (value_type != dtype_) {
        return Internal("value type of queue does not match specified Batcher value type");
      }

      return Status::OK();
    }

    Status DequeueElement(OpKernelContext *ctx, bool &emitted) {
      Notification n;
      auto s = Status::OK();
      input_queue_->TryDequeue(ctx, [&](const QueueInterface::Tuple &tuple) {
        auto &record_id_t = tuple[0];
        auto &record_id = record_id_t.scalar<string>()();
        if (current_record_id_.empty()) {
          current_record_id_ = record_id;
        } else if (record_id != current_record_id_) {
          s.Update(EnqueueOutput(ctx));
          if (s.ok()) {
            current_record_id_ = record_id;
            emitted = true;
          }
        }

        if (s.ok()) {
          batch_.push_back(move(tuple[1]));
        }
        emitted = false;
        n.Notify();
      });
      n.WaitForNotification();
      return s;
    }

    Status EnqueueOutput(OpKernelContext *ctx) {
      TensorShape out_shape(expected_shape_), scalar_shape({});
      auto batch_size = batch_.size();
      out_shape.InsertDim(0, batch_size);

      Tensor *batch_out_t, *request_id_out_t;
      TF_RETURN_IF_ERROR(ctx->allocate_output("batched_tensor", out_shape, &batch_out_t));
      TF_RETURN_IF_ERROR(ctx->allocate_output("request_id", scalar_shape, &request_id_out_t));
      request_id_out_t->scalar<string>()() = current_record_id_;
      // BEGIN needs to be templated
      // You can use the templating stuff. see the polymorphism section here: https://www.tensorflow.org/extend/adding_an_op#polymorphism
      auto batch_out_parent = batch_out_t->flat_outer_dims<string>(); // FIXME use the stuff like in CopySliceToElement from QueueBase (with the macro)
      for (size_t i = 0; i < batch_size; i++) {
        batch_out_parent.chip(i, 0) = batch_[i].flat<string>();
      }
      // END needs to be templated

      return Status::OK();
    }

  };
  REGISTER_KERNEL_BUILDER(Name("Batcher").Device(DEVICE_CPU), BatcherOp);
} // namespace tensorflow {
