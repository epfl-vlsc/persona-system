
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include "tensorflow/core/framework/queue_interface.h"
#include <vector>
#include <cstdint>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"

namespace tensorflow {

   namespace { 
      void resource_releaser(ResourceContainer<Data> *data) {
        core::ScopedUnref a(data);
        data->release();
      }
   }


  using namespace std;
  using namespace errors;
  using namespace format;

  class AGDProteinClusterOp : public OpKernel {
  public:
    AGDProteinClusterOp(OpKernelConstruction *context) : OpKernel(context) {
      // ring ID is compared to sequence ID of incoming chunks to see if we have processed it before
      // indicating it has made a full trip around the ring
      OP_REQUIRES_OK(context, context->GetAttr("ring_size", &ring_size_));

      // should seed allows this node to seed a cluster if it has none
      // only one op in the ring should_seed to more closely replicate single thread results
      OP_REQUIRES_OK(context, context->GetAttr("should_seed", &should_seed_));
    }

    Status Init(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &input_queue_));
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 1), &neighbor_queue_));
    }


    void Compute(OpKernelContext* ctx) override {
      LOG(INFO) << "Starting protein cluster";
      if (!bufferpair_pool_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      /*Tensor chunk_t, num_recs_t, seq_number_t, was_added_t;
      bool done_chunk = false;
      while (!done_chunk) {

        OP_REQUIRES_OK(ctx, DequeueChunk(ctx, chunk_t, num_recs_t, sequence_t, was_added_t));
        auto sequence = sequence_t.scalar<int32>();
        int32 new_sequence = sequence + 1;
        
        auto chunk = chunk_t.scalar<string>();
        auto num_seqs = num_seqs_t.scalar<int32>();
        AGDRecordReader seqs_reader(chunk.data(), num_seqs);

        const char * data;
        size_t len;
        Status s = seqs_reader.GetNextRecord(&data, &len);
        OP_REQUIRES_OK(ctx, s);

        if (sequence == ring_size_) {
          // this chunk has been evaluated by all nodes
          // create new clusters for any non added proteins

        }
        
        if (clusters_.empty() && should_seed_) {
          // seed a new cluster with first sequence
        } else {
          // pass this chunk to neighbor
        }

        while (s.ok()) {
          for (auto& cluster : clusters_) {
            cluster.EvaluateSequence
        }
      }*/
      
      

      //LOG(INFO) << "DONE running mark duplicates!! Found so far: " << num_dups_found_;

    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;
    QueueInterface *input_queue_ = nullptr;
    QueueInterface *neighbor_queue_ = nullptr;
    //vector<ProteinCluster> clusters_;
  
    int ring_size_;
    bool should_seed_ = false;
      
    inline Status DequeueChunk(OpKernelContext* ctx, Tensor& chunk, Tensor& num_recs, Tensor& sequence, Tensor& was_added) {
        Notification n;
          input_queue_->TryDequeue(ctx, [this, &n, &chunk, &num_recs, &sequence, 
              &was_added](const QueueInterface::Tuple &tuple) {
              chunk = tuple[0];
              num_recs = tuple[1];
              sequence = tuple[2];
              was_added = tuple[3];
              n.Notify();
          });
        n.WaitForNotification();
        return Status::OK();
      }
      
    inline void EnqueueChunk(OpKernelContext *ctx, const Tensor& chunk, 
        const Tensor& num_recs, const Tensor& sequence, const Tensor& was_added) {
        Notification n;
        QueueInterface::Tuple tuple;
        tuple.reserve(4);
        tuple[0] = chunk; tuple[1] = num_recs;
        tuple[2] = sequence; tuple[3] = was_added;
        neighbor_queue_->TryEnqueue(tuple, ctx, [&n]() {
            n.Notify();
        });
        n.WaitForNotification();
      }

  };

  REGISTER_KERNEL_BUILDER(Name("AGDProteinCluster").Device(DEVICE_CPU), AGDProteinClusterOp);
} //  namespace tensorflow {
