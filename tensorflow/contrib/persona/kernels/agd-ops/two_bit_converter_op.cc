#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/agd-format/shared_mmap_file_resource.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/parser.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"

namespace tensorflow {

  using namespace std;
  using namespace errors;

  class TwoBitConverterOp : public OpKernel {
  public:
    TwoBitConverterOp(OpKernelConstruction *context) : OpKernel(context) {
    }

    void Compute(OpKernelContext* ctx) override {
      ResourceContainer<Data> *input_data;
      int32_t num_records;
      OP_REQUIRES_OK(ctx, GetInputData(ctx, &input_data, &num_records));
      core::ScopedUnref a(input_data); // NOTE: do not release this buffer. it is passed downstreams still
      auto data = input_data->get();
      OP_REQUIRES_OK(ctx, ConvertToTwoBit(*data, num_records));
      OP_REQUIRES_OK(ctx, input_data->allocate_output("output", ctx));
    }

  private:

    inline
    Status GetInputData(OpKernelContext *ctx, ResourceContainer<Data> **input_data, int32_t *num_records) {
      const Tensor *input_data_t, *num_records_t;
      TF_RETURN_IF_ERROR(ctx->input("input", &input_data_t));
      TF_RETURN_IF_ERROR(ctx->input("num_records", &num_records_t));
      auto input_data_handle = input_data_t->vec<string>();
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(input_data_handle(0), input_data_handle(1), input_data));
      *num_records = num_records_t->scalar<int32>()();
      return Status::OK();
    }

    inline
    Status ConvertToTwoBit(Data &buffer, int32_t num_records) {
      auto data = buffer.mutable_data();
      if (data == nullptr) {
        return Internal("2bit Converter: mutable data unavailable for 2-bit converter");
      }

      // We assume that we get a valid buffer from the converter
      // Turn on verify in the AGDReader op if you want this
      auto index = reinterpret_cast<const format::RelativeIndex*>(data);
      auto record_data = data + num_records * sizeof(format::RelativeIndex);
      size_t offset = 0, record_size;
      char result, ascii_base;

      for (decltype(num_records) i = 0; i < num_records; ++i) {
        record_size = index[i];
        for (decltype(record_size) record_idx = 0; record_idx < record_size; ++record_idx, ++offset) {
          ascii_base = record_data[offset];
          result = nst_nt4_table[ascii_base];
          if (result > 4) {
            return Internal("Got an invalid result from 2bit conversion for ", string(1, ascii_base), ": ", (int) result);
          }
          record_data[offset] = result;
        }
      }

      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("TwoBitConverter").Device(DEVICE_CPU), TwoBitConverterOp);
} // namespace tensorflow {
