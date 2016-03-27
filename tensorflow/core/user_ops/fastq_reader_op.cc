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

// Reader OP to read FASTQ files and returns 
#include <memory>
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/kernels/reader_base.h"
#include "tensorflow/core/user_ops/dna-align/snap_proto.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("FastqReader")
    .Output("reader_handle: Ref(string)")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .Doc(R"doc(
A Reader that outputs the read sequences in a FASTQ file. Does not output
score values or metadata. 

reader_handle: The handle to reference the Reader.
skip_header_lines: Number of lines to skip from the beginning of every file.
container: If non-empty, this reader is placed in the given container.
        Otherwise, a default container is used.
)doc");

class FastqReader : public ReaderBase {
 public:
  FastqReader(const string& node_name, int skip_header_lines, Env* env)
      : ReaderBase(strings::StrCat("FastqReader '", node_name, "'")),
        skip_header_lines_(skip_header_lines),
        env_(env),
        line_number_(0) {}

  Status OnWorkStartedLocked() override {
    line_number_ = 0;
    RandomAccessFile* file = nullptr;
    TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file));
    input_buffer_.reset(new io::InputBuffer(file, kBufferSize));
    for (; line_number_ < skip_header_lines_; ++line_number_) {
      string line_contents;
      Status status = input_buffer_->ReadLine(&line_contents);
      if (errors::IsOutOfRange(status)) {
        // We ignore an end of file error when skipping header lines.
        // We will end up skipping this file.
        return Status::OK();
      }
      TF_RETURN_IF_ERROR(status);
    }
    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    input_buffer_.reset(nullptr);
    return Status::OK();
  }

  Status ReadLocked(string* key, string* value, bool* produced,
                    bool* at_end) override {

    //LOG(INFO) << "Reading from file " << current_work() << " !\n";

    string lines[4];
    Status status;
    bool atend = false;
    for (int i = 0; i < 4; i++)
    {
      status = input_buffer_->ReadLine(&lines[i]);
      if (i != 3 && errors::IsOutOfRange(status)) {
        errors::Internal("FASTQ Read error in " + current_work() +
                " : EOF encountered in read.");
      } else if (errors::IsOutOfRange(status)) {
        // just EOF
        atend = true;
      }
      ++line_number_;
    }

    if (status.ok()) {
        SnapProto::AlignmentDef alignment;
        SnapProto::ReadDef* read = alignment.mutable_read();
        read->set_bases(lines[1]);
        read->set_meta(lines[0]);
        read->set_length(lines[1].length());
        read->set_qualities(lines[3]);
        alignment.SerializeToString(value);
        *key = strings::StrCat(current_work(), ":", line_number_);
        *produced = true;
        *at_end = atend;
        return status;
    } else {
        return status;
    }

  }

  Status ResetLocked() override {
    line_number_ = 0;
    input_buffer_.reset(nullptr);
    return ReaderBase::ResetLocked();
  }

 private:
  enum { kBufferSize = 256 << 10 /* 256 kB */ };
  const int skip_header_lines_;
  Env* const env_;
  int64 line_number_;
  int64 batch_size_;
  std::unique_ptr<io::InputBuffer> input_buffer_;
};

class FastqReaderOp : public ReaderOpKernel {
 public:
  explicit FastqReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    int skip_header_lines = -1;
    int read_batch_size = -1;
    OP_REQUIRES_OK(context,
                   context->GetAttr("skip_header_lines", &skip_header_lines));
    OP_REQUIRES_OK(context,
                   context->GetAttr("read_batch_size", &read_batch_size));
    OP_REQUIRES(context, skip_header_lines >= 0,
                errors::InvalidArgument("skip_header_lines must be >= 0 not ",
                                        skip_header_lines));
    Env* env = context->env();
    SetReaderFactory([this, skip_header_lines, read_batch_size, env]() {
      return new FastqReader(name(), skip_header_lines, env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("FastqReader").Device(DEVICE_CPU),
                        FastqReaderOp);

}  // namespace tensorflow
