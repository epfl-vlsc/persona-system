#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/user_ops/dense-format/buffer.h"
#include "tensorflow/core/user_ops/dense-format/format.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "GenomeIndex.h"
#include "genome_index_resource.h"
#include "Read.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/SAM.h"
#include <cstdint>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  REGISTER_OP("DenseTester")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("sam_filename: string = ''")
  .Input("num_records: int32")
  .Input("dense_records: string")
  .Input("genome_handle: Ref(string)")
  .Output("num_records_out: int32")
  .Output("dense_records_out: string")
  .Doc(R"doc(
  Compares the dense format output with the SAM format output
)doc");

  class DenseTesterOp : public OpKernel {
  public:
    
    DenseTesterOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("sam_filename", &sam_filename_));

      genome_resource_ = NULL;
    }
		
    _int64 getFileSize(const char *fileName) {
      struct stat sb;
      int fd = open(fileName, O_RDONLY);
      int r = fstat(fd, &sb);
      _int64 fileSize = sb.st_size;
      close(fd);
      return fileSize;
    }

    void Compute(OpKernelContext* ctx) override {

      // Get the dense format results in dense_records
      // rec_data(0) = container, rec_data(1) = name
    	ResourceContainer<Data> *records;
      const Tensor *rec_input;
      OP_REQUIRES_OK(ctx, ctx->input("dense_records", &rec_input));
      auto rec_input_vec = rec_input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(rec_input_vec(0), rec_input_vec(1), &records));
      auto rec_data = records->get()->data();

      // Get number of records per chunk
      const Tensor *tensor;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &tensor));
      auto num_records = tensor->scalar<int32>()();

            // Outputs the "dense_records" input
      const TensorShape rec_handle_shape({2});
      Tensor *rec_handle;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("dense_records_out", rec_handle_shape, &rec_handle));
      auto rec_handle_vec = rec_handle->vec<string>();
      rec_handle_vec(0) = rec_input_vec(0);
      rec_handle_vec(1) = rec_input_vec(1); 

      // Outputs the "num_records" input
      Tensor *num_rec_handle;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records_out", TensorShape(), &num_rec_handle));
      auto num_rec_out = num_rec_handle->template scalar<int32>();
      num_rec_out() = num_records;
    
      if (NULL == genome_resource_)
      {
        // One-time init
        OP_REQUIRES_OK(ctx,
          GetResourceFromContext(ctx, "genome_handle", &genome_resource_));
        const Genome *genome = genome_resource_->get_genome();

        // Populate a context necessary for reading the SAM file
        readerContext_.genome = genome;
        readerContext_.ignoreSecondaryAlignments = true;
        readerContext_.ignoreSupplementaryAlignments = true;
        readerContext_.clipping = NoClipping;
        readerContext_.defaultReadGroup = "";
        readerContext_.header = NULL;
        readerContext_.headerLength = 0;
        readerContext_.headerBytes = 0;

        _int64 sam_filesize = getFileSize(sam_filename_.c_str());
        reader_ = SAMReader::create(DataSupplier::Default, sam_filename_.c_str(), 2, readerContext_, 0, sam_filesize);
      }

      // Will be populated in SAMReader::getNextRead
      AlignmentResult sam_alignmentResult;
      GenomeLocation sam_genomeLocation;
      Direction sam_direction; 
      unsigned sam_mapQ;
      unsigned sam_flag;
      const char *sam_cigar;

      const char *curr_record = rec_data + num_records; // skip the indices

      for (int i = 0; i < num_records; i++) {
        if (! reader_->getNextRead(&sam_read_, &sam_alignmentResult, &sam_genomeLocation, 
            &sam_direction, &sam_mapQ, &sam_flag, &sam_cigar)) {
          std::cout << "Could not get read from SAM file!" << std::endl;
        }
        
        size_t record_size = static_cast<size_t>(*(rec_data + i)); // indices are stored as char
        size_t var_string_size = record_size - sizeof(format::AlignmentResult);

        const format::AlignmentResult *dense_result = 
              reinterpret_cast<const format::AlignmentResult *>(curr_record);
        std::string dense_cigar(reinterpret_cast<const char*>(curr_record + sizeof(format::AlignmentResult)), var_string_size);

        if (sam_genomeLocation != dense_result->location_) {
          std::cout << "For record " << i << " the SAM location is " << sam_genomeLocation 
              << " and the dense location is " << dense_result->location_ << std::endl;
        }
        
        if (sam_mapQ != dense_result->mapq_) {
          std::cout << "For record " << i << " the SAM mapQ is " << sam_mapQ 
              << " and the dense mapQ is " << dense_result->mapq_ << std::endl;
        }

        if (sam_flag != dense_result->flag_) {
          std::cout << "For record " << i << " the SAM flag is " << sam_flag 
              << " and the dense flag is " << dense_result->flag_ << std::endl;
        }

//        if (dense_cigar.compare(sam_cigar)) {
//          std::cout << "For record " << i << " the SAM cigar is " << sam_cigar 
//              << " and the dense flag is " << dense_cigar << std::endl;
//        }

        curr_record = curr_record + record_size;
      }
    }

  private:     
    string sam_filename_;
    GenomeIndexResource* genome_resource_;
    ReaderContext readerContext_;
    SAMReader *reader_;
    Read sam_read_;
  };


REGISTER_KERNEL_BUILDER(Name("DenseTester").Device(DEVICE_CPU), DenseTesterOp);
} // namespace tensorflow
