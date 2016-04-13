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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/kernels/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "format.h"
#include "decompress.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

namespace tensorflow {

#define REGISTER_DENSE_READER(_name_)                   \
  REGISTER_OP(_name_)                                   \
    .Output("reader_handle: Ref(string)")         \
    .Attr("container: string = ''")               \
    .Attr("shared_name: string = ''")             \
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

class DenseReader : public ReaderBase {
public:
  DenseReader(const string& node_name, Env* env)
    : ReaderBase(strings::StrCat("DenseReader '", node_name, "'")),
      env_(env), ordinal_start_(0), current_idx_(0), records_(nullptr), record_count_(0),
      current_record_(nullptr) {};

  Status OnWorkStartedLocked() override {
    using namespace std;

    if (!records_) {
      ReadOnlyMemoryRegion *mapped_file = nullptr;
      TF_RETURN_IF_ERROR(env_->NewReadOnlyMemoryRegionFromFile(current_work(), &mapped_file));
      std::unique_ptr<ReadOnlyMemoryRegion> close_for_sure(mapped_file);

      const uint64_t file_size = mapped_file->length();

      if (file_size < sizeof(format::FileHeader)) {
        return errors::ResourceExhausted("Dense file '", current_work(), "' is not large enough for file header. Actual size ",
                                         file_size, " bytes is less than necessary size of ",
                                         sizeof(format::FileHeader), " bytes");
      }

      auto file_data = reinterpret_cast<const char*>(mapped_file->data());
      auto file_header = reinterpret_cast<const format::FileHeader*>(file_data);
      auto payload_start = reinterpret_cast<const char*>(file_data + file_header->segment_start);
      const size_t payload_size = file_size - file_header->segment_start;

      Status status;
      if (static_cast<format::CompressionType>(file_header->compression_type) == format::CompressionType::BZIP2) {
        status = decompressBZIP2(payload_start, payload_size, output_);
      } else {
        status = copySegment(payload_start, payload_size, output_);
      }
      TF_RETURN_IF_ERROR(status);

      // 2. do math on the file size
      const uint64_t num_records = file_header->last_ordinal - file_header->first_ordinal; // FIXME off-by-1?
      const auto output_size = output_.size();
      if (output_size < num_records) {
        output_.clear();
        return errors::InvalidArgument("Mapped file '", current_work(), "' is smaller than the stated index entries");
      }

      auto data = output_.data();
      auto records = reinterpret_cast<const format::RecordTable*>(data);
      size_t data_size = 0;
      for (uint64_t i = 0; i < num_records; ++i) {
        data_size += records->relative_index[i];
      }

      const size_t expected_size = output_size - num_records;
      if (data_size != expected_size) {
        if (data_size < expected_size) {
          output_.clear();
          return errors::OutOfRange("Expected a file size of ", expected_size, " bytes, but only found",
                                    data_size, " bytes in ", current_work());
        } else {
          LOG(WARNING) << current_work() << " has an extra " << data_size - expected_size << " bytes in uncompressed format";
        }
      }

      records_ = records;
      record_count_ = num_records;
      ordinal_start_ = file_header->first_ordinal;
      // skip the record table. If no records, ReadLocked will catch it
      current_record_ = data + num_records;
    }

    current_idx_ = 0;

    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    current_idx_ = 0;
    return Status::OK();
  }

  Status ResetLocked() override {
    current_idx_ = 0;
    current_record_ = output_.data() + record_count_;
    return ReaderBase::ResetLocked();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    using namespace std;

    const char* record;
    size_t record_length;

    if (GetCurrentRecord(&record, &record_length)) {
      *value = string(record, record_length);
      *key = strings::StrCat(current_work(), ":", current_idx_, "-", ordinal_start_+current_idx_);
      *produced = true;
      AdvanceRecord();
    } else {
      *at_end = true;
    }

    return Status::OK();
  }

protected:

  bool GetCurrentRecord(const char** record, std::size_t *record_length) {
    if (current_idx_ < record_count_) {
      *record = current_record_;
      *record_length = records_->relative_index[current_idx_];
      return true;
    } else {
      return false;
    }
  }

  void AdvanceRecord() {
    if (current_idx_ < record_count_) {
      current_record_ += records_->relative_index[current_idx_++];
    }
  }

private:
  Env* const env_;
  std::vector<char> output_;
  const format::RecordTable* records_;
  const char* current_record_;

  // Some indices for keeping track of our ordinals
protected:
  std::uint64_t ordinal_start_;
  std::uint64_t current_idx_;
  std::size_t record_count_;
};

class BaseReader : public DenseReader {
public:
  BaseReader(const string& node_name, Env *env) :
    DenseReader(node_name, env) {}


  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    using namespace std;
    using namespace format;
    const char* record;
    size_t record_length;

    if (GetCurrentRecord(&record, &record_length)) {
      auto bases = reinterpret_cast<const BinaryBaseRecord*>(record);
      TF_RETURN_IF_ERROR(bases->toString(record_length, value));
      *key = strings::StrCat(current_work(), ":", current_idx_, "-", ordinal_start_+current_idx_);
      *produced = true;
      AdvanceRecord();
    } else {
      *at_end = true;
    }

    return Status::OK();
  }
};

template <typename T>
class DenseReaderOp : public ReaderOpKernel {
public:
  explicit DenseReaderOp(OpKernelConstruction* context)
    : ReaderOpKernel(context) {

    Env* env = context->env();
    SetReaderFactory([this, env]() {
        return new T(name(), env);
      });
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU),
                        DenseReaderOp<DenseReader>);

REGISTER_KERNEL_BUILDER(Name("BaseReader").Device(DEVICE_CPU),
                        DenseReaderOp<BaseReader>);

} // namespace tensorflow
