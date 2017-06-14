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
//#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"
//#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
//#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/Bam.h"
//#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/Read.h"
//#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/ReadSupplierQueue.h"
#include "zlib.h"

// Required to read the SRA file
#include <ncbi-vdb/NGS.hpp>
#include <ngs/ErrorMsg.hpp>
#include <ngs/ReadCollection.hpp>
#include <ngs/ReadIterator.hpp>
#include <iostream>
#include <string>

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;
  using namespace ngs;

  using shape_inference::InferenceContext;
   
/*  namespace { 
    void resource_releaser(ResourceContainer<Data> *data) {
      core::ScopedUnref a(data);
      data->release();
    }
  }*/

  class AgdImportSraOp : public OpKernel {
    //ReadIterator iterator;
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
        /*OP_REQUIRES_OK(ctx, ctx->GetAttr("unaligned", &unaligned_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ref_seq_lens", &ref_seq_lens_));

        // build a dummy genome so SNAP BAMreader will give us correct
        // location information from an alignment
        genome_.reset(new Genome(0xffffffff, 0xffffffff, 0));

        ref_size_totals_.reserve(ref_seq_lens_.size());
        int64 total = 0;
        vector<char> dummy;
        for (auto len : ref_seq_lens_) {
          genome_->startContig("test");
          dummy.resize(len);
          genome_->addData(&dummy[0], len);

          total += len;
          //ref_vec.push_back(RefData(ref_seqs_[i], ref_sizes_[i]));
          ref_size_totals_.push_back(total);
        }
        genome_->fillInContigLengths();

        reader_context_.genome = genome_.get();
        reader_ = new BAMReader(reader_context_);
        reader_->init(path.c_str(), ReadSupplierQueue::BufferCount(num_threads), 0, 0);
*/
	// initialize the Sra Reader
	cout << "path is " << path.c_str() << "\n"; 
	ReadCollection reader = ncbi::NGS::openReadCollection ( path.c_str() );
	numReads = reader.getReadCount();
	cout << "reads" << numReads << "\n";
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
        //delete reader_;
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
     //   if (unaligned_) {
          OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk_out", TensorShape({3, 2}), &out_t));
       // } else {
         // OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk_out", TensorShape({4, 2}), &out_t));
       // }

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
        // if paired, num_recs_out will be 2x chunksize
        int num_recs_out;
        //if (unaligned_) {
          s = Process(base_builder, qual_builder, meta_builder, num_recs_out);
       // } else {
         // AlignmentResultBuilder results_builder;
         // OP_REQUIRES_OK(ctx, GetBufferForBuilder(ctx, results_builder, out_t, 3));
         // s = ProcessAligned(base_builder, qual_builder, meta_builder, results_builder, num_recs_out);
        //}
        num_recs = num_recs_out;
        first_ordinal_ += num_recs_out;
        OP_REQUIRES_OK(ctx, s);
      }

    private:

      vector<BinaryBases> bases_;

      Status Process(ColumnBuilder& bases, ColumnBuilder& qual, ColumnBuilder& meta, int& num_recs) {
//	ReadIterator iterator = reader.getReadRange(1, numReads);
        num_recs = 0;
     //   cout << "num recs before is " << num_recs;
	
        for (size_t i = 0; i < chunk_size_; i++) {
	   if (!iterator->nextRead()) {
   	      return OutOfRange("No more reads in the SRA file");	
	   }
	  // char base[251];
	   //char quality[251];
	   if (iterator->getReadBases().size() > 250) {
             /// base = base.substr(0,250);
	      //char* base;
	      //char* quality;
	      //strncpy(base, iterator->getReadBases().data(), 250);
	      //strncpy(quality, iterator->getReadQualities().data(), 250);
	      //base[250] = '\0';
	      //quality[250] = '\0';
	      if (i % 1000000 == 0){
//                 cout << "default size should only print i = " << i <<" since chunk size is" << chunk_size_;   
              }
	      TF_RETURN_IF_ERROR(IntoBases(iterator->getReadBases().data(), 250, bases_));

              bases.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
              qual.AppendRecord(iterator->getReadQualities().data(), 250);

		 //   quality = quality.substr(0,250); 
	   } else {
             TF_RETURN_IF_ERROR(IntoBases(iterator->getReadBases().data(), iterator->getReadBases().size(), bases_));

             bases.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
             qual.AppendRecord(iterator->getReadQualities().data(), iterator->getReadQualities().size());
	   }
          //TF_RETURN_IF_ERROR(IntoBases(base, base.size(), bases_));

           //bases.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
           //qual.AppendRecord(quality, quality.size());
	   meta.AppendRecord(iterator->getReadName().data(), iterator->getReadName().size());
           num_recs++;

	}
//	cout << "num recs after is " << num_recs;
	return Status::OK();
      }

      Status ProcessUnaligned(ColumnBuilder& bases, ColumnBuilder& qual, ColumnBuilder& meta, int& num_recs) {
      /*  Read bam_read;
        AlignmentResult result;
        GenomeLocation location;
        bool isRC;
        unsigned mapq;
        unsigned flag;
        const char* cigar;
        num_recs = 0;
        for (size_t i = 0; i < chunk_size_; i++) {
          if (!reader_->getNextRead(&bam_read, &result, &location, &isRC, &mapq, &flag, 
                &cigar)) {
            return OutOfRange("No more reads in BAM file");
          }

          TF_RETURN_IF_ERROR(IntoBases(bam_read.getUnclippedData(), bam_read.getUnclippedLength(), bases_));

          bases.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
          qual.AppendRecord(bam_read.getUnclippedQuality(), bam_read.getUnclippedLength());
          meta.AppendRecord(bam_read.getId(), bam_read.getIdLength());
          num_recs++;
        }*/
        return Status::OK();
      }
      
      /*Status ProcessAligned(ColumnBuilder& bases, ColumnBuilder& qual, ColumnBuilder& meta,
          AlignmentResultBuilder& results, int& num_recs) {
        Read bam_read, mate_bam_read;
        AlignmentResult result;
        GenomeLocation location, mate_location;
        bool isRC, mate_isRC;
        unsigned mapq;
        unsigned flag;
        const char* cigar;
        Alignment alignment, mate_alignment;
        int64 position;
        int32 ref_idx;
        int64 mate_position;
        int32 mate_ref_idx;
        num_recs = 0;
        for (size_t i = 0; i < chunk_size_; i++) {
          if (!reader_->getNextRead(&bam_read, &result, &location, &isRC, &mapq, &flag, 
                &cigar)) {
            return OutOfRange("No more reads in BAM file");
          }
          if (IsSecondary(flag) || IsSupplemental(flag)) {// skip secondary/supplemental
            i--;
            continue;
          }
          if (isRC) {
            bam_read.becomeRC();
          }
          TF_RETURN_IF_ERROR(IntoBases(bam_read.getUnclippedData(), bam_read.getUnclippedLength(), bases_));

          bases.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
          qual.AppendRecord(bam_read.getUnclippedQuality(), bam_read.getUnclippedLength());
          meta.AppendRecord(bam_read.getId(), bam_read.getIdLength());
          position = FindChromosome(location, ref_idx);
          alignment.mutable_position()->set_position(position);
          alignment.mutable_position()->set_ref_index(ref_idx);
          alignment.set_mapping_quality(mapq);
          alignment.set_flag(flag);
          alignment.set_cigar(cigar);
          alignment.set_template_length(0);
          num_recs++;
          
          if (IsPaired(flag)) {
            if (!reader_->getNextRead(&mate_bam_read, &result, &mate_location, &mate_isRC, &mapq, &flag, 
                  &cigar)) {
              return Internal("pair not found for read!!");
            }
            if (mate_isRC) {
              mate_bam_read.becomeRC();
            }
            TF_RETURN_IF_ERROR(IntoBases(mate_bam_read.getData(), mate_bam_read.getDataLength(), bases_));

            bases.AppendRecord(reinterpret_cast<const char*>(&bases_[0]), sizeof(BinaryBases)*bases_.size());
            qual.AppendRecord(mate_bam_read.getUnclippedQuality(), mate_bam_read.getUnclippedLength());
            meta.AppendRecord(mate_bam_read.getId(), mate_bam_read.getIdLength());
            mate_position = FindChromosome(mate_location, mate_ref_idx);
            mate_alignment.mutable_position()->set_position(mate_position);
            mate_alignment.mutable_position()->set_ref_index(mate_ref_idx);
            mate_alignment.set_mapping_quality(mapq);
            mate_alignment.set_flag(flag);
            mate_alignment.set_cigar(cigar);
           
            // deal with paired alignment stuff, some of this copied from SNAP
            alignment.mutable_next_position()->set_position(mate_position);
            alignment.mutable_next_position()->set_ref_index(mate_ref_idx);
            mate_alignment.mutable_next_position()->set_position(position);
            mate_alignment.mutable_next_position()->set_ref_index(ref_idx);
            int64 basesClippedBefore, basesClippedAfter;
            int64 clippedLength = bam_read.getDataLength();
            int64 fullLength = bam_read.getUnclippedLength();
            if (isRC) {
              basesClippedBefore = fullLength - clippedLength - bam_read.getFrontClippedLength();
              basesClippedAfter = bam_read.getFrontClippedLength();
            } else {
              basesClippedBefore = bam_read.getFrontClippedLength();
              basesClippedAfter = fullLength - clippedLength - basesClippedBefore;
            }
            GenomeLocation myStart = location - basesClippedBefore;
            GenomeLocation myEnd = location + clippedLength + basesClippedAfter;
            int64 mateBasesClippedBefore = mate_bam_read.getFrontClippedLength();
            int64 mateBasesClippedAfter = mate_bam_read.getUnclippedLength() - mate_bam_read.getDataLength() - mateBasesClippedBefore;
            GenomeLocation mateStart = mate_location - (mate_isRC == RC ? mateBasesClippedAfter : mateBasesClippedBefore);
            GenomeLocation mateEnd = mate_location + mate_bam_read.getDataLength() + (!mate_isRC ? mateBasesClippedAfter : mateBasesClippedBefore);
            int template_length;
            if (ref_idx == mate_ref_idx) { 
              if (myStart < mateStart) {
                template_length = mateEnd - myStart;
              } else {
                template_length = -(myEnd - mateStart);
              }
            } // otherwise leave TLEN as zero.
            alignment.set_template_length(template_length);
            mate_alignment.set_template_length(-template_length);
            results.AppendAlignmentResult(alignment);
            results.AppendAlignmentResult(mate_alignment);

            num_recs++;
          } else
            results.AppendAlignmentResult(alignment);

        }
        return Status::OK();
      }*/

      ReferencePool<BufferPair> *bufpair_pool_ = nullptr;

     /* int64 FindChromosome(int64 location, int &ref_idx) {
        int index = 0;
        while (location > ref_size_totals_[index]) index++;
        ref_idx = index;
        return (index == 0) ? (int)location : (int)(location - ref_size_totals_[index-1]);
      }*/

     // unique_ptr<Genome> genome_;
     // bool unaligned_;
  //    BAMReader* reader_ = nullptr;
    //  ReaderContext reader_context_; 
      int chunk_size_;
      ReadIterator* iterator = nullptr;
      int64 first_ordinal_ = 0;
    //  vector<int> ref_seq_lens_;
     // vector<int64> ref_size_totals_;;
      TF_DISALLOW_COPY_AND_ASSIGN(AgdImportSraOp);
  };


  REGISTER_KERNEL_BUILDER(Name("AgdImportSra").Device(DEVICE_CPU), AgdImportSraOp);

}  // namespace tensorflow
