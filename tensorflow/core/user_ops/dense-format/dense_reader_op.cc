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

class DenseReader : public ReaderBase {
public:
  DenseReader(const string& node_name, Env* env)
    : ReaderBase(strings::StrCat("DenseReader '", node_name, "'")),
      env_(env), ordinal_start_(0), current_idx_(0), record_count_(0) {};

  Status OnWorkStartedLocked() override {
    using namespace std;
    namespace bios = boost::iostreams;

    bios::mapped_file_source input_dense_file;
    input_dense_file.open(current_work());

    if (!input_dense_file.is_open()) {
      return errors::Unavailable("Cannot create mapped file of '", current_work(), "'");
    }

    const size_t file_size = input_dense_file.size();

    if (file_size < sizeof(format::FileHeader)) {
      return errors::ResourceExhausted("Dense file '", current_work(), "' is not large enough for file header. Actual size ",
                                       file_size, " bytes is less than necessary size of ",
                                       sizeof(format::FileHeader), " bytes");
    }

    const char* file_data = input_dense_file.data();
    const format::FileHeader *file_header = reinterpret_cast<const format::FileHeader*>(file_data);
    const char* payload_start = file_data + file_header->segment_start;
    const size_t payload_size = file_size - file_header->segment_start;

    Status status;
    if (static_cast<format::CompressionType>(file_header->compression_type) == format::CompressionType::BZIP2) {
      status = decompressBZIP2(payload_start, payload_size, output_);
    } else {
      status = copySegment(payload_start, payload_size, output_);
    }
    TF_RETURN_IF_ERROR(status);

    // 2. do math on the file size
    
    // 3. read in the rest of the file size, but use boost stuff (memory mapping) instead of slow system calls and unneeded buffering

    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    return Status::OK();
  }

  Status ResetLocked() override {
    return ReaderBase::ResetLocked();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {
    return Status::OK();
  }

private:
  Env* const env_;
  std::vector<char> output_;

  // Some indices for keeping track of our ordinals
  std::uint64_t ordinal_start_;
  std::uint64_t current_idx_;
  std::size_t record_count_;
};

class DenseReaderOp : public ReaderOpKernel {
  public:
    explicit DenseReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {

      Env* env = context->env();
      SetReaderFactory([this, env]() {
          return new DenseReader(name(), env);
        });
    }
  };

REGISTER_KERNEL_BUILDER(Name("DenseReader").Device(DEVICE_CPU),
                        DenseReaderOp);

} // namespace tensorflow
