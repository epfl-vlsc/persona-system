/* @author Alaleh Azhir
*  SRA import op
*  Output chunks of bases, qual, metadata
*  Using NCBI library to read the SRA file
*/

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include <tuple>
#include <thread>
#include <vector>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"
#include "zlib.h"

// Required to read the SRA file
#include <ncbi-vdb/NGS.hpp>
#include <ngs/ErrorMsg.hpp>
#include <ngs/ReadCollection.hpp>
#include <ngs/ReadIterator.hpp>
#include <iostream>

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;
  using namespace ngs;

  using shape_inference::InferenceContext;

  class AgdImportSraOp : public OpKernel {
    unsigned int numReads;
    public:
      // This Op WILL throw OutOfRange upon Sra file read completion, 
      // downstream ops should catch this (use a queue)
      explicit AgdImportSraOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
   
        string path;
        int num_threads = 0;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("path", &path));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
	
        // initialize the Sra Reader
	ReadCollection reader = ncbi::NGS::openReadCollection ( path.c_str() );
	numReads = reader.getReadCount();
	cout << "Number of reads: " << numReads << "\n";
	iterator = new ReadIterator(reader.getReadRange(1, numReads));
      }
    
      Status GetOutputBufferPair(OpKernelContext* ctx, ResourceContainer<BufferPair> **ctr)
      {
        TF_RETURN_IF_ERROR(bufpair_pool_->GetResource(ctr));
        (*ctr)->get()->reset();
        return Status::OK();
      }

      Status InitHandles(OpKernelContext* ctx)
      {
        TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "bufpair_pool", &bufpair_pool_));

        return Status::OK();
      }

      ~AgdImportSraOp() override {
        // drain the queues first
        core::ScopedUnref unref_listpool(bufpair_pool_);
	delete iterator;
      }

      Status GetBufferForBuilder(OpKernelContext* ctx, ColumnBuilder& builder, Tensor* out, int index) {
        ResourceContainer<BufferPair> *output_bufpair_ctr;
        TF_RETURN_IF_ERROR(GetOutputBufferPair(ctx, &output_bufpair_ctr));
        auto output_bufferpair = output_bufpair_ctr->get();
        builder.SetBufferPair(output_bufferpair);
        auto out_mat = out->matrix<string>();
        out_mat(index, 0) = output_bufpair_ctr->container();
        out_mat(index, 1) = output_bufpair_ctr->name();
        return Status::OK();
      }

      void Compute(OpKernelContext* ctx) override {
       	if (!bufpair_pool_) {
          OP_REQUIRES_OK(ctx, InitHandles(ctx));
        }

        Tensor* out_t, *num_recs_t, *first_ord_t;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk_out", TensorShape({3, 2}), &out_t));
      	OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records", TensorShape({}), &num_recs_t));
        OP_REQUIRES_OK(ctx, ctx->allocate_output("first_ordinal", TensorShape({}), &first_ord_t));

        auto& num_recs = num_recs_t->scalar<int>()();
        auto& first_ord = first_ord_t->scalar<int64>()();
        first_ord = first_ordinal_;

        ColumnBuilder base_builder;
        OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, base_builder, out_t, 0));
        ColumnBuilder qual_builder;
        OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, qual_builder, out_t, 1));
        ColumnBuilder meta_builder;
        OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, meta_builder, out_t, 2));

        Status s;
        int num_recs_out;
        s = Process(base_builder, qual_builder, meta_builder, num_recs_out);
        num_recs = num_recs_out;
        first_ordinal_ += num_recs_out;
        OP_REQUIRES_OK(ctx, s);
      }

    private:

      vector<BinaryBases> bases_;

      Status Process(ColumnBuilder& bases, ColumnBuilder& qual, ColumnBuilder& meta, int& num_recs) {
        num_recs = 0;
        for (size_t i = 0; i < chunk_size_; i++) {
	   if (!iterator->nextRead()) {
   	      return OutOfRange("No more reads in the SRA file");	
	   }
	   if (iterator->getReadBases().size() > UINT16_MAX) {
              return Internal("An error occured"); 
	   } 
           TF_RETURN_IF_ERROR(IntoBases(iterator->getReadBases().data(), iterator->getReadBases().size(), bases_));
           bases.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
           qual.AppendRecord(iterator->getReadQualities().data(), iterator->getReadQualities().size());
	   meta.AppendRecord(iterator->getReadName().data(), iterator->getReadName().size());
           num_recs++;
	}
	return Status::OK();
      }

      ReferencePool<BufferPair> *bufpair_pool_ = nullptr;

      int chunk_size_;
      ReadIterator* iterator = nullptr;
      int64 first_ordinal_ = 0;
      TF_DISALLOW_COPY_AND_ASSIGN(AgdImportSraOp);
  };


  REGISTER_KERNEL_BUILDER(Name("AgdImportSra").Device(DEVICE_CPU), AgdImportSraOp);

}  
