#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "shared_mmap_file_resource.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "format.h"
#include "decompress.h"
#include <vector>

namespace tensorflow {

  REGISTER_OP("DenseReader")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("batch_size: int")
  .Input("file_set_handle: string")
  .Output("buffer_handle: string")
  .SetIsStateful()
  .Doc(R"doc(
Reads the dense stuff
  )doc");

  using namespace std;

  class DenseReader : public OpKernel {
  public:
    DenseReader(OpKernelConstruction *context) : OpKernel(context) {
      using namespace errors;
      int batch_size;
      OP_REQUIRES_OK(context, context->GetAttr("batch_size",
                                               &batch_size));
      batch_size_ = batch_size;
      OP_REQUIRES(context, batch_size > 0, InvalidArgument("DenseReaderOp: batch_size must be >0 - ", batch_size));
    }

    void Compute(OpKernelContext* ctx) override {
      using namespace errors;
      const Tensor *fileset;
      OP_REQUIRES_OK(ctx, ctx->input("file_set_handle", &fileset));
      OP_REQUIRES(ctx, fileset->shape() == TensorShape({3,2}), InvalidArgument("Tensorshape is incorrect for dense reader op"));

      auto fileset_mat = fileset->matrix<string>();
      const auto &base_container = fileset_mat(0,0);
      const auto &base_name = fileset_mat(0,1);
      const auto &quality_container = fileset_mat(1,0);
      const auto &quality_name = fileset_mat(1,1);
      const auto &metadata_container = fileset_mat(2,0);
      const auto &metadata_name = fileset_mat(2,1);
    }
  protected:
    Status ReadStandardRecord(string *result, const char* record, const size_t length) {
      *result = string(record, length);
      return Status::OK();
    }

    Status ReadBaseRecord(string *result, const char* record, const size_t length) {
      auto bases = reinterpret_cast<const format::BinaryBaseRecord*>(record);
      return bases->toString(length, result);
    }

    Status
    ProcessFile(MemoryMappedFile &mmf, OpKernelContext *ctx)
    {
      using namespace format;
      using namespace errors;

      vector<char> buffer;

      auto mem_map_file = mmf.get();
      const auto data = mem_map_file->data();
      const auto char_data = reinterpret_cast<const char*>(data);
      auto length = mem_map_file->length();

      if (length < sizeof(FileHeader)) {
        return Internal("DenseReader::FillBuffer: needed min length ", sizeof(FileHeader),
                        ", but only received ", length);
      }
      auto file_header = reinterpret_cast<const FileHeader*>(data);

      auto record_type = static_cast<RecordType>(file_header->record_type);
      switch (record_type) {
      default:
        return Internal("Invalid record type", file_header->record_type);
      case RecordType::BASES:
      case RecordType::QUALITIES:
      case RecordType::COMMENTS:
        break;
      }

      auto payload_start = char_data + file_header->segment_start;
      auto payload_size = length - file_header->segment_start;
      auto data_start_idx = file_header->segment_start;

      Status status;
      if (static_cast<CompressionType>(file_header->compression_type) == CompressionType::BZIP2) {
        status = decompressBZIP2(payload_start, payload_size, buffer);
      } else {
        status = copySegment(payload_start, payload_size, buffer);
      }
      TF_RETURN_IF_ERROR(status);

      const uint64_t index_size = file_header->last_ordinal - file_header->first_ordinal;
      if (buffer.size() - data_start_idx < index_size * 2) {
        return Internal("FillBuffer: expected at least ", index_size*2, " bytes, but only have ", buffer.size() - data_start_idx);
      } else if (index_size > batch_size_) {
        return Internal("FillBuffer: decompressed a chunk with ", index_size, " elements, but maximum batch size is ", batch_size_);
      }

      auto records = reinterpret_cast<const RecordTable*>(&buffer[data_start_idx]);
      size_t data_size = 0;
      for (uint64_t i = 0; i < index_size; ++i) {
        data_size += records->relative_index[i];
      }

      const size_t expected_size = buffer.size() - (data_start_idx + index_size);
      if (data_size != expected_size) {
        if (data_size < expected_size) {
          return errors::OutOfRange("Expected a file size of ", expected_size, " bytes, but only found",
                                    data_size, " bytes");
        } else {
          return errors::OutOfRange("Expected a file size of ", expected_size, " bytes, but only found",
                                    data_size, " bytes");
        }
      }

      //allocate output
      Tensor *value = nullptr;
      TF_RETURN_IF_ERROR(ctx->allocate_output("value", TensorShape({batch_size_}), &value));
      auto flat = value->vec<string>();
      int i = 0;
      uint8_t current_record_len;
      const char* current_record = payload_start + index_size;
      string parsed;
      for (; i < index_size; i++) {
        current_record_len = records->relative_index[i];
        if (record_type == RecordType::BASES) {
          TF_RETURN_IF_ERROR(ReadBaseRecord(&parsed, current_record, current_record_len));
        } else {
          TF_RETURN_IF_ERROR(ReadStandardRecord(&parsed, current_record, current_record_len));
        }
        current_record += current_record_len;
        flat(i) = parsed;
      }

      for (; i < batch_size_; i++) {
        flat(i) = "";
      }

      return Status::OK();
    }
  private:
    int batch_size_;
  };

  REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU), DenseReader);
} //  namespace tensorflow {
