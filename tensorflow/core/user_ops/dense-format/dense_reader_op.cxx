/* Copyright 2015 Google Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   ==============================================================================*/

#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/kernels/reader_async_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "format.h"
#include "decompress.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

namespace tensorflow {

#define REGISTER_DENSE_READER(_name_)                   \
  REGISTER_OP(_name_)                                   \
  .Output("reader_handle: Ref(string)")                 \
  .Attr("container: string = ''")                       \
  .Attr("shared_name: string = ''")                     \
  .Attr("batch_size: int")                              \
  .Attr("parallel: int")                                \
  .Attr("buffer: int")                                  \
  .SetIsStateful()

  REGISTER_DENSE_READER("DenseReader")
    .Doc(R"doc(
    A Reader that outputs the records from the DenseFormat in our custom file.
    Only outputs a single type. To use with the aligner, a downstream aggregator node is needed.

    reader_handle: The handle to reference the Reader.
    container: If non-empty, this reader is placed in the given container.
    Otherwise, a default container is used.
    shared_name: a name for a shared resource
    )doc");

  REGISTER_DENSE_READER("BaseReader")
    .Doc(R"doc(
    A Reader that outputs the records from *bases* the DenseFormat in our custom file.
    Only outputs a single type. To use with the aligner, a downstream aggregator node is needed.

    reader_handle: The handle to reference the Reader.
    container: If non-empty, this reader is placed in the given container.
    Otherwise, a default container is used.
    shared_name: a name for a shared resource
    )doc");

using namespace std;

class DenseReader : public ReaderAsyncBase {
public:
  DenseReader(Env* env, int batch_size, int parallel, int buffer)
    : ReaderAsyncBase(parallel, buffer),
      env_(env), data_buf_(nullptr), batch_size_(batch_size)
  {
    if (batch_size_ < 1) {
      LOG(ERROR) << "batch size is non-positive!: " << batch_size_;
    }
  }
  TensorShape GetRequiredShape() override
  {
    // Just an array of the strings of individual reads
    return TensorShape({batch_size_});
  }

  Status FillBuffer(InputChunk *chunk, std::vector<char> &buffer)
  {
    using namespace format;
    using namespace errors;

    buffer.clear();
    const void *data = nullptr;
    size_t length = 0;
    chunk->GetChunk(&data, &length);
    if (length < sizeof(FileHeader)) {
      return Internal("DenseReader::FillBuffer: needed min length ", sizeof(FileHeader),
                      ", but only received ", length);
    }
    auto char_data = reinterpret_cast<const char*>(data);
    auto filename = chunk->GetFileName();
    copy(filename.begin(), filename.end(), back_inserter(buffer));
    buffer.push_back('\0');
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

    buffer.insert(buffer.end(), char_data, char_data + file_header->segment_start);

    auto payload_start = char_data + file_header->segment_start;
    auto payload_size = length - file_header->segment_start;
    size_t data_start_idx = buffer.size();

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
                                  data_size, " bytes in ", filename);
      } else {
        return errors::OutOfRange("Expected a file size of ", expected_size, " bytes, but only found",
                                  data_size, " bytes in ", filename);
      }
    }

    return Status::OK();
  }

  Status ChunkWorkItem(const string &filename)
  {
    using namespace std;
    ReadOnlyMemoryRegion *mapped_file = nullptr;
    TF_RETURN_IF_ERROR(env_->NewReadOnlyMemoryRegionFromFile(filename, &mapped_file));
    shared_ptr<ReadOnlyMemoryRegion> file_resource(mapped_file);
    InputChunk chunk(file_resource, 0, mapped_file->length());
    chunk.SetFileName(filename);
    return EnqueueNextChunk(std::move(chunk));
  }

  Status ParseFileNameAndData(const vector<char>& buffer, string *filename, const char** data, size_t *size)
  {
    string filename_tmp(&buffer[0]); // goes up until NULL termination
    *filename = move(filename_tmp);
    *data = &buffer[filename->size()+1]; // size() does not include null termination, so +1 to skip over it
    *size = buffer.size() - filename->size()+1;
    return Status::OK();
  }

  Status ReadStandardRecord(string *result, const char* record, const size_t length) {
    *result = string(record, length);
    return Status::OK();
  }

  Status ReadBaseRecord(string *result, const char* record, const size_t length) {
    auto bases = reinterpret_cast<const format::BinaryBaseRecord*>(record);
    return bases->toString(length, result);
  }

  Status ReadBatchLocked(Tensor* batch_tensor, string *key, int* num_produced, bool *done_with_buffer) override
  {
    using namespace format;
    using namespace errors;
    const vector<char>* current_buf = nullptr;
    if (!GetCurrentBuffer(&current_buf)) {
      return errors::Unavailable("DenseReader::ReadBatchLocked:: Unable to get current buffer");
    }

    string filename;
    const char *data;
    size_t data_len;
    TF_RETURN_IF_ERROR(ParseFileNameAndData(*current_buf, &filename, &data, &data_len));

    auto file_header = reinterpret_cast<const FileHeader*>(data);
    auto payload_start = data + file_header->segment_start;
    auto records = reinterpret_cast<const RecordTable*>(payload_start);
    auto record_type = static_cast<RecordType>(file_header->record_type);
    uint64_t index_len = file_header->last_ordinal - file_header->first_ordinal;

    auto batch_vec = batch_tensor->vec<string>();

    string parsed;
    const char* current_record = payload_start + index_len;
    uint8_t current_record_len;
    for (uint64_t i = 0; i < index_len; i++) {
      current_record_len = records->relative_index[i];
      if (record_type == RecordType::BASES) {
        TF_RETURN_IF_ERROR(ReadBaseRecord(&parsed, current_record, current_record_len));
      } else {
        TF_RETURN_IF_ERROR(ReadStandardRecord(&parsed, current_record, current_record_len));
      }
      current_record += current_record_len;
      batch_vec(i) = parsed;
    }

    if (index_len < batch_size_) {
      uint64_t padding_elems = batch_size_ - index_len;
      LOG(DEBUG) << "Inserting " << padding_elems << " padding elements in batch";
      for (uint64_t i = index_len; i < batch_size_; i++) {
        batch_vec(i) = "";
      }
    }

    *key = filename;

    *num_produced = index_len;
    *done_with_buffer = true;
    return Status::OK();
  }

private:
  Env* const env_;
  const vector<char> *data_buf_;
  const int batch_size_;
};

class DenseReaderOp : public ReaderOpKernel {
public:
  explicit DenseReaderOp(OpKernelConstruction* context)
    : ReaderOpKernel(context) {
    using namespace errors;
    int batch_size;
    OP_REQUIRES_OK(context, context->GetAttr("batch_size",
                                             &batch_size));
    OP_REQUIRES(context, batch_size > 0, InvalidArgument("DenseReaderOp: batch_size must be >0 - ", batch_size));
    int parallel;
    OP_REQUIRES_OK(context, context->GetAttr("parallel",
                                             &parallel));
    OP_REQUIRES(context, parallel > 0, InvalidArgument("DenseReaderOp: parallel must be >0 - ", parallel));
    int buffer;
    OP_REQUIRES_OK(context, context->GetAttr("buffer",
                                             &buffer));
    OP_REQUIRES(context, buffer > 0, InvalidArgument("DenseReaderOp: buffer must be >0 - ", buffer));
    Env* env = context->env();
    SetReaderFactory([this, env, batch_size, parallel, buffer]() {
        return new DenseReader(env, batch_size, parallel, buffer);
      });
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU), DenseReaderOp);

} // namespace tensorflow
