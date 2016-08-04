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
  .Input("genome_handle: Ref(string)")
  .Input("dense_records: string")
  .Input("num_records: int32")
  .Doc(R"doc(
  Compares the dense format output with the SAM format output
)doc");

  class DenseTesterOp : public OpKernel {
  public:

    DenseTesterOp(OpKernelConstruction* context) : OpKernel(context)
    {
      OP_REQUIRES_OK(context, context->GetAttr("sam_filename", &sam_filename_));
    }


    void Compute(OpKernelContext* ctx) override {
      if (!genome_resource_) {
        OP_REQUIRES_OK(ctx, init(ctx));
      }

      // Get the dense format results in dense_records
      // rec_data(0) = container, rec_data(1) = name
      ResourceContainer<Data> *records;
      const Tensor *rec_input;
      OP_REQUIRES_OK(ctx, ctx->input("dense_records", &rec_input));
      auto rec_input_vec = rec_input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(rec_input_vec(0), rec_input_vec(1), &records));
      auto rec_data = records->get()->data();


      // Get number of records per chunk
      const Tensor *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->scalar<int32>()();

      // Will be populated in SAMReader::getNextRead
      AlignmentResult sam_alignmentResult;
      GenomeLocation sam_genomeLocation;
      Direction sam_direction;
      unsigned sam_mapQ;
      unsigned sam_flag;
      const char *sam_cigar;
      size_t num_char, record_size, var_string_size;
      auto size_index = reinterpret_cast<const format::RecordTable*>(rec_data);

      const char *curr_record = rec_data + num_records; // skip the indices
      const format::AlignmentResult *dense_result;
      bool should_error = false;

      auto status = Status::OK();
      for (decltype(num_records) i = 0; i < num_records; ++i) {
        if (!reader_->getNextRead(&sam_read_, &sam_alignmentResult, &sam_genomeLocation,
                                  &sam_direction, &sam_mapQ, &sam_flag, &sam_cigar)) {
          LOG(INFO) << "Could not get read from SAM file!\n";
          status = Internal( "Could not get read from SAM file!\n");
          break;
        }

        record_size = size_index->relative_index[i];
        var_string_size = record_size - sizeof(format::AlignmentResult);

        dense_result = reinterpret_cast<const format::AlignmentResult *>(curr_record);

        string dense_cigar(reinterpret_cast<const char*>(curr_record + sizeof(format::AlignmentResult)), var_string_size);

        if (sam_genomeLocation != dense_result->location_) {
          LOG(INFO) << "Mismatch: for record " << i + 1 << " the SAM location is " << sam_genomeLocation
              << " and the dense location is " << dense_result->location_ << "\n";
          should_error = true;
        }

        if (sam_mapQ != dense_result->mapq_) {
          LOG(INFO) << "Mismatch: for record " << i + 1 << " the SAM mapQ is " << sam_mapQ
              << " and the dense mapQ is " << dense_result->mapq_ << "\n";
          should_error = true;
        }

        if (sam_flag != dense_result->flag_) {
          LOG(INFO) << "Mismatch: for record " << i + 1 << " the SAM flag is " << sam_flag
              << " and the dense flag is " << dense_result->flag_ << "\n";
          should_error = true;
        }

        // sam_cigar is not null terminated
        num_char = static_cast<size_t>(strchr(sam_cigar, '\t') - sam_cigar);
        string sam_cigar_end(sam_cigar, num_char);

        if (dense_cigar.compare(sam_cigar_end)) {
          LOG(INFO) << "Mismatch: for record " << i + 1 << " the SAM cigar is " << sam_cigar_end
              << " and the dense cigar is " << dense_cigar << "\n";
          should_error = true;
        }

        curr_record += record_size;
      }

      if (should_error) {
        status = Internal("Dense record set did not pass verification. Please enable LOG(INFO) for more details");
      }

      OP_REQUIRES_OK(ctx, status);
    }

  private:

    Status getFileSize(const string &filename, size_t *file_size) {
      struct stat stat_buf;
      int r = stat(filename.c_str(), &stat_buf);
      if (r != 0) {
        return Internal("fstat on file ", filename, " returned error code ", r);
      }
      *file_size = stat_buf.st_size;
      return Status::OK();
    }

    Status init(OpKernelContext *ctx)
    {
      // One-time init
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "genome_handle", &genome_resource_));
      auto const *genome = genome_resource_->get_genome();

      // Populate a context necessary for reading the SAM file
      readerContext_.genome = genome;
      readerContext_.ignoreSecondaryAlignments = true;
      readerContext_.ignoreSupplementaryAlignments = true;
      readerContext_.clipping = NoClipping;
      readerContext_.defaultReadGroup = "";
      readerContext_.header = NULL;
      readerContext_.headerLength = 0;
      readerContext_.headerBytes = 0;

      size_t sam_filesize;
      TF_RETURN_IF_ERROR(getFileSize(sam_filename_, &sam_filesize));
      reader_ = SAMReader::create(DataSupplier::Default, sam_filename_.c_str(), 2, readerContext_, 0, sam_filesize);
      return Status::OK();
    }

    string sam_filename_;
    GenomeIndexResource* genome_resource_ = nullptr;
    ReaderContext readerContext_;
    SAMReader *reader_ = nullptr;
    Read sam_read_;
  };


REGISTER_KERNEL_BUILDER(Name("DenseTester").Device(DEVICE_CPU), DenseTesterOp);
} // namespace tensorflow
