#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <vector>
#include <cstdint>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"
#include "tensorflow/core/framework/queue_interface.h"
#include <boost/functional/hash.hpp>
#include <google/dense_hash_map>
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"

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

  inline bool operator==(const Position& lhs, const Position& rhs) {
    return (lhs.ref_index() == rhs.ref_index() && lhs.position() == rhs.position());
  }

  class AGDPrintMultiOp : public OpKernel {
  public:
    AGDPrintMultiOp(OpKernelConstruction *context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("ref_sequences", &ref_seqs_));
        OP_REQUIRES_OK(context, context->GetAttr("ref_seq_sizes", &ref_sizes_));
        LOG(INFO) << "Started printing " ;
        //count_=0;

        // flag is used to generate output array only for those references which have some alignment in the read chunks
    }

    ~AGDPrintMultiOp()
    {
      core::ScopedUnref a1(input_queue_);
      LOG(INFO) << "Done Printing " ;
    }


    // main function to increment the counter for each result in the agd column

    void Compute(OpKernelContext* ctx) override {
      cout << "started" <<endl;
      fflush(stdout);
      if (!input_queue_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }
      cout << "queue initialized" <<endl;

      int no =0;

      Status s = DequeueElement(ctx);

      while(s.ok()){

        cout << "dequeued chunk" << ++no << endl;
        fflush(stdout);
        s = DequeueElement(ctx);

      }


      Tensor *out_t;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({3, 2}), &out_t));

      // done
    }

    Status Init(OpKernelContext *ctx) {
      TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, 0), &input_queue_));

      // auto &input_dtypes = input_queue_->component_dtypes();
      // if (!(input_dtypes.at(0) == DT_STRING)) {
      //   return Internal("Barrier: input or output queue has a non-string type for first element");
      // }
      return Status::OK();
    }

    Status LoadDataResource(OpKernelContext* ctx, const Tensor* handle_t,ResourceContainer<Data>** container) {
      auto rmgr = ctx->resource_manager();
      auto handles_vec = handle_t->vec<string>();

      TF_RETURN_IF_ERROR(rmgr->Lookup(handles_vec(0), handles_vec(1), container));
      return Status::OK();
    }

    Status DequeueElement(OpKernelContext *ctx) {
      Notification n;
      int sf = -1;
      //Status s;
      input_queue_->TryDequeue(ctx, [&](const QueueInterface::Tuple &tuple) {
          //out << input_queue_->size();

        //cout << tuple.size() << endl;
        if(tuple.size()==0)
        {
          sf = 2;
          n.Notify();
        }
        else
        {

          base_t = &tuple[0];
          // auto &base = base_t.scalar<string>()();
          quality_t = &tuple[1];
          // auto &quality = quality_t.scalar<string>()();
          result_t = &tuple[2];
          // auto &result = result_t.scalar<string>()();
          num_records_t = &tuple[3];
          // auto &num_records = num_records_t.scalar<string>()();

          auto num_records = num_records_t->scalar<int32>()();
          ResourceContainer<Data>* bases_data, *qual_data, *result_data;
          OP_REQUIRES_OK(ctx, LoadDataResource(ctx, base_t, &bases_data));
          OP_REQUIRES_OK(ctx, LoadDataResource(ctx, quality_t, &qual_data));
          OP_REQUIRES_OK(ctx, LoadDataResource(ctx, result_t, &result_data));
          AGDRecordReader base_reader(bases_data, num_records);
          AGDRecordReader qual_reader(qual_data, num_records);
          AGDResultReader results_reader(result_data, num_records);

          print_alignment(&results_reader);
          //newBaseReader = &base_reader;
          //newQualReader = &qual_reader;
          //*newResultReader = results_reader;


          n.Notify();
        }

      });
      n.WaitForNotification();
      if(sf==2)
      {
        return NotFound("reached end of file");
      }
      return Status :: OK();
    }

    void print_alignment(AGDResultReader *results_reader){
      Alignment result;
      Status s = results_reader->GetNextResult(result);
      //cout << result.position().position()<<endl;
      int recs =0;
      // this detection logic adapted from SamBlaster
      while (s.ok()) {
        //cout << "started5" <<endl;
        if (!IsPrimary(result.flag()))
          //OP_REQUIRES_OK(ctx, Internal("Non-primary result detected in primary result column at location ",
            //    result.position().DebugString()));

          // we have a single alignment
          if (IsUnmapped(result.flag())) {
            //cout << "isUnmapped";
            fflush(stdout);
            s = results_reader->GetNextResult(result);
            continue;
          }
        //cout << result.position().ref_index() << " "<< result.position().position() << " "<<result.cigar() << endl;
        s = results_reader->GetNextResult(result);
        recs++;
        //cout << "reading a chunk of file" << endl;
        fflush(stdout);
      }
      //cout << recs << endl;
    }


  private:
    vector<string> ref_seqs_;
    vector<int32> ref_sizes_;
    QueueInterface *input_queue_ = nullptr;
    const Tensor *result_t, *base_t, *quality_t, *num_records_t;
    // for various command line arguments


  };

  REGISTER_KERNEL_BUILDER(Name("AGDPrintMulti").Device(DEVICE_CPU), AGDPrintMultiOp);
} //  namespace tensorflow {
