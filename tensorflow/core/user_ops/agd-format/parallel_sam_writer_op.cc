#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "data.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "compression.h"
#include "buffer_list.h"
#include "format.h"
#include "util.h"
#include "read_resource.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/AlignerContext.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/SingleAligner.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/AlignmentResult.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/FileFormat.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/SAM.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/Read.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/DataWriter.h"
#include "tensorflow/core/user_ops/dna-align/genome_index_resource.h"
#include "tensorflow/core/user_ops/dna-align/aligner_options_resource.h"

#define SAM_REVERSE_COMPLEMENT 0x10
#define NUM_ARGS 5

namespace tensorflow {
  using namespace std;
  using namespace errors;
  namespace {
    const string op_name("ParallelSamWriter");
  }

  REGISTER_OP(op_name.c_str())
  // TODO: check to see what attributes I still need
  .Attr("sam_file_path: string = ''")
  .Input("agd_results: string")
  .Input("genome_handle: Ref(string)")
  .Input("options_handle: Ref(string)")
 // TODO these can be collapsed into a vec(3) if that would help performance
	.Input("read: string")
  .Input("num_records: int32")
  .Output("num_records_out: int32")
  .SetIsStateful()
  .Doc(R"doc(
Writes out the output in the SAM format (just a character buffer) to the location specified by the input.

This writes out to local disk only

Assumes that the record_id for a given set does not change for the runtime of the graph
and is thus passed as an Attr instead of an input (for efficiency);
)doc");

  class ParallelSamWriterOp : public OpKernel {
  public:
 		ParallelSamWriterOp(OpKernelConstruction *ctx) : OpKernel(ctx), argc_(NUM_ARGS), argv_{"single", "/scratch/stuart/ref_index", "/scratch/genome/one_mil.fastq", "-o", "output.sam"} {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("sam_file_path", &sam_file_path_));
    } 

		void Compute(OpKernelContext *ctx) override {
			if (!writer_supplier_) {
				OP_REQUIRES_OK(ctx, init(ctx));
			}		
			
			// Get the agd format results in agd_records
      ResourceContainer<BufferList> *records;
      const Tensor *rec_input;
      OP_REQUIRES_OK(ctx, ctx->input("agd_results", &rec_input));
      auto rec_input_vec = rec_input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(rec_input_vec(0), rec_input_vec(1), &records));
      core::ScopedUnref column_releaser(records);
      auto rec_data_list_p = records->get();
      auto& rec_data_list = *rec_data_list_p;
      rec_data_list.wait_for_ready();
      
			auto num_buffers = rec_data_list.size();
      LOG(INFO) << "num buffers " << num_buffers;

			// Get number of records per chunk
      const Tensor *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->scalar<int32>()();
      LOG(INFO) << "num records: " << num_records;
      uint32_t records_per_chunk = num_records / num_buffers;
      if (num_records % num_buffers != 0) {
        ++records_per_chunk;
      }
      LOG(INFO) << "records per chunk" << records_per_chunk;

      // Get the reads corresponding to the results
			ResourceContainer<ReadResource> *reads_container;
			const Tensor *read_input;
			OP_REQUIRES_OK(ctx, ctx->input("read", &read_input));
			auto data_read = read_input->vec<string>(); // data_read(0) = container, data_read(1) = name
			OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(data_read(0), data_read(1), &reads_container));
			auto reads = reads_container->get();

      auto status = Status::OK();

			int cur_buflist_index = 0;
			Buffer *index = &rec_data_list[cur_buflist_index].index();
			auto size_index = reinterpret_cast<const format::RecordTable*>(&(*index)[0]);
      size_t size_index_size = index->size();
      size_t cur_size_index = 0;
      Buffer* data = &rec_data_list[cur_buflist_index].data();
      const char *curr_record = data->data(); // skip the indices
      size_t record_size;

      const format::AlignmentResult *agd_result;

      const char *bases, *qualities;
      size_t bases_len, qualities_len;

      Read snap_read;
            
      for (decltype(num_records) i = 0; i < num_records; ++i) {

        status = reads->get_next_record(&bases, &bases_len, &qualities, &qualities_len);
        if (!status.ok()) {
         LOG(INFO) << "Failed to get next read";
         return; 
        }

        snap_read.init(nullptr, 0, bases, qualities, bases_len);
        snap_read.clip(options_->clipping);

				if ((snap_read.getDataLength() < options_->minReadLength || snap_read.countOfNs() > options_->maxDist) && (!options_->passFilter(&snap_read, AlignmentResult::NotFound, true, false))) {
            LOG(INFO) << "FILTERING READ";
            continue;
        }

				record_size = size_index->relative_index[cur_size_index];
        agd_result = reinterpret_cast<const format::AlignmentResult *>(curr_record);
      
        // convert format::AlignmentResult to SingleAlignmentResult
        SingleAlignmentResult result_for_sam;
        result_for_sam.location = agd_result->location_;
        result_for_sam.score = agd_result->score_;
        result_for_sam.mapq = agd_result->mapq_;
        result_for_sam.direction = (agd_result->flag_ & SAM_REVERSE_COMPLEMENT) ? RC : FORWARD; 
        result_for_sam.status = (agd_result->location_ == InvalidGenomeLocation) ? static_cast<AlignmentResult>(0) : static_cast<AlignmentResult>(1);

        read_writer_->writeReads(reader_context_, &snap_read, &result_for_sam, 1, true);

				if (cur_size_index == size_index_size - 1 && i != num_records - 1) {
          cur_buflist_index++;
          index = &rec_data_list[cur_buflist_index].index();
          size_index = reinterpret_cast<const format::RecordTable*>(&(*index)[0]);
          size_index_size = index->size();
          cur_size_index = 0;
          data = &rec_data_list[cur_buflist_index].data();
          curr_record = data->data(); // skip the indices
        } else {
          cur_size_index++;
          curr_record += record_size;
        }
      }
      
      Tensor *num_recs;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records_out", TensorShape({}), &num_recs));
      num_recs->scalar<int32>()() = num_records;
		}

    ~ParallelSamWriterOp() override {
      if (read_writer_ != NULL) {
        read_writer_->close();
        delete read_writer_;
      }

      core::ScopedUnref index_unref(genome_resource_);
      core::ScopedUnref options_unref(options_resource_);
    }

		Status init(OpKernelContext *ctx) {
			// One-time init
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "genome_handle", &genome_resource_));
      genome_ = genome_resource_->get_genome();

      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "options_handle", &options_resource_));
      options_ = options_resource_->value();
      
      memset(&reader_context_, 0, sizeof(reader_context_));
			reader_context_.clipping = options_->clipping;
			reader_context_.defaultReadGroup = options_->defaultReadGroup;
			reader_context_.genome = genome_; 
			reader_context_.ignoreSecondaryAlignments = options_->ignoreSecondaryAlignments;
			reader_context_.ignoreSupplementaryAlignments = options_->ignoreSecondaryAlignments;   // Maybe we should split them out
			DataSupplier::ExpansionFactor = options_->expansionFactor;

      const FileFormat *format = FileFormat::SAM[options_->useM];
			
			format->setupReaderContext(options_, &reader_context_);

			dataSupplier = DataWriterSupplier::create(sam_file_path_.c_str(), options_->writeBufferSize);
      writer_supplier_ = ReadWriterSupplier::create(format, dataSupplier, genome_);
      ReadWriter *headerWriter = writer_supplier_->getWriter();
      headerWriter->writeHeader(reader_context_, options_->sortOutput, argc_, argv_, snap_version_, options_->rgLineContents, options_->outputFile.omitSQLines);
      headerWriter->close();
      delete headerWriter;

      read_writer_ = writer_supplier_->getWriter();

      return Status::OK();
		}
  
  private:
    GenomeIndexResource *genome_resource_ = nullptr;
    AlignerOptionsResource *options_resource_ = nullptr;
  	const Genome *genome_;
    string sam_file_path_;
		AlignerOptions *options_; // Keep all the options in one place, similar to SNAP
    ReaderContext reader_context_;
    DataWriterSupplier *dataSupplier;
  	ReadWriterSupplier *writer_supplier_ = nullptr;
		ReadWriter *read_writer_ = nullptr;

    const int argc_;
    const char *argv_[NUM_ARGS];
    const char *snap_version_ = "1.0beta.23";
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), ParallelSamWriterOp);
}
