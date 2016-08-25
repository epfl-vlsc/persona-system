#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/user_ops/agd-format/buffer_list.h"
#include "tensorflow/core/user_ops/agd-format/format.h"
#include "tensorflow/core/user_ops/agd-format/util.h"
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

  REGISTER_OP("AGDTester")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("sam_filename: string = ''")
  .Input("genome_handle: Ref(string)")
  .Input("agd_records: string")
  .Input("num_records: int32")
  .Output("agd_records_out: string")
  .Output("num_records_out: int32")
  .Doc(R"doc(
  Compares the agd format output with the SAM format output
)doc");

  class AGDTesterOp : public OpKernel {
  public:

    AGDTesterOp(OpKernelConstruction* context) : OpKernel(context)
    {
      OP_REQUIRES_OK(context, context->GetAttr("sam_filename", &sam_filename_));
    }

    void Compute(OpKernelContext* ctx) override {
      if (!genome_resource_) {
        OP_REQUIRES_OK(ctx, init(ctx));
      }

      // Get the agd format results in agd_records
      // rec_data(0) = container, rec_data(1) = name
      ResourceContainer<BufferList> *records;
      const Tensor *rec_input;
      OP_REQUIRES_OK(ctx, ctx->input("agd_records", &rec_input));
      auto rec_input_vec = rec_input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(rec_input_vec(0), rec_input_vec(1), &records));
      core::ScopedUnref column_releaser(records);
      auto &rec_data_list = records->get()->get_when_ready();
      auto num_buffers = rec_data_list.size();

      // Get number of records per chunk
      const Tensor *num_records_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
      auto num_records = num_records_t->scalar<int32>()();
      uint32_t records_per_chunk = num_records / num_buffers;
      if (num_records % num_buffers != 0) {
        ++records_per_chunk;
      }

      auto status = Status::OK();
      decltype(num_records) i = 0, recs_per_chunk = records_per_chunk;
      buf_.clear();
      for (auto &buffer : rec_data_list) {
        // deals with the last chunk, which could have a smaller number of records
        // depending on the params
        if (i + recs_per_chunk > num_records) {
          recs_per_chunk = num_records - i;
        }
        auto &data_buf = buffer.get(); // only need to do this on the first call
        OP_REQUIRES_OK(ctx, appendSegment(&data_buf[0], recs_per_chunk, buf_, true));
        i += recs_per_chunk;
      }

      i = 0; recs_per_chunk = records_per_chunk;
      size_t expected_size;
      for (auto &buffer : rec_data_list) {
        if (i + recs_per_chunk > num_records) {
          recs_per_chunk = num_records - i;
        }
        auto &data_buf = buffer.get();
        expected_size = data_buf.size() - recs_per_chunk;
        OP_REQUIRES_OK(ctx, appendSegment(&data_buf[recs_per_chunk], expected_size, buf_, true));
        i += recs_per_chunk;
      }

      // Will be populated in SAMReader::getNextRead
      AlignmentResult sam_alignmentResult;
      GenomeLocation sam_genomeLocation;
      Direction sam_direction;
      unsigned sam_mapQ;
      unsigned sam_flag;
      const char *sam_cigar;
      size_t num_char, record_size, var_string_size;

      auto size_index = reinterpret_cast<const format::RecordTable*>(&buf_[0]);

      const char *curr_record = &buf_[num_records]; // skip the indices
      const format::AlignmentResult *agd_result;
      bool should_error = false;

      for (decltype(num_records) i = 0; i < num_records; ++i) {
        if (!reader_->getNextRead(&sam_read_, &sam_alignmentResult, &sam_genomeLocation,
                                  &sam_direction, &sam_mapQ, &sam_flag, &sam_cigar)) {
          LOG(INFO) << "Could not get read from SAM file!\n";
          status = Internal( "Could not get read from SAM file!\n");
          break;
        }

        record_size = size_index->relative_index[i];
        var_string_size = record_size - sizeof(format::AlignmentResult);

        agd_result = reinterpret_cast<const format::AlignmentResult *>(curr_record);

        string agd_cigar(reinterpret_cast<const char*>(curr_record + sizeof(format::AlignmentResult)), var_string_size);

        if (sam_genomeLocation != agd_result->location_) {
          LOG(INFO) << "Mismatch: for record " << i + 1 << " the SAM location is " << sam_genomeLocation
              << " and the agd location is " << agd_result->location_ << "\n";
          should_error = true;
        }

        if (sam_mapQ != agd_result->mapq_) {
          LOG(INFO) << "Mismatch: for record " << i + 1 << " the SAM mapQ is " << sam_mapQ
              << " and the agd mapQ is " << agd_result->mapq_ << "\n";
          should_error = true;
        }

        if (sam_flag != agd_result->flag_) {
          LOG(INFO) << "Mismatch: for record " << i + 1 << " the SAM flag is " << sam_flag
              << " and the agd flag is " << agd_result->flag_ << "\n";
          should_error = true;
        }

        // sam_cigar is not null terminated
        num_char = static_cast<size_t>(strchr(sam_cigar, '\t') - sam_cigar);
        string sam_cigar_end(sam_cigar, num_char);

        if (agd_cigar.compare(sam_cigar_end)) {
          LOG(INFO) << "Mismatch: for record " << i + 1 << " the SAM cigar is " << sam_cigar_end
              << " and the agd cigar is " << agd_cigar << "\n";
          should_error = true;
        }

        curr_record += record_size;
      }

      if (should_error) {
        status = Internal("AGD record set did not pass verification. Please enable LOG(INFO) for more details");
      } else {
        Tensor *agd_out, *records_out;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("agd_records_out", rec_input->shape(), &agd_out));
        OP_REQUIRES_OK(ctx, ctx->allocate_output("num_records_out", num_records_t->shape(), &records_out));
        *agd_out = *rec_input;
        *records_out = *num_records_t;
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
    vector <char> buf_;
  };


REGISTER_KERNEL_BUILDER(Name("AGDTester").Device(DEVICE_CPU), AGDTesterOp);
} // namespace tensorflow
