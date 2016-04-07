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
      FastqReader(const string& node_name, Env* env)
        : ReaderBase(strings::StrCat("FastqReader '", node_name, "'")),
        env_(env),
        line_number_(0) {}

      Status OnWorkStartedLocked() override {
        line_number_ = 0;
        RandomAccessFile* file = nullptr;
        LOG(INFO) << "Opening file: " << current_work() << std::endl;
        TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(current_work(), &file));
        input_buffer_.reset(new io::InputBuffer(file, kBufferSize));
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
        for (int i = 0; i < 4; i++)
        {
          status = input_buffer_->ReadLine(&lines[i]);
          ++line_number_;
          if (errors::IsOutOfRange(status)) {
            LOG(INFO) << "I is out of range! file: " 
              << current_work() << " line number: " << line_number_ << std::endl;
            *at_end = true;
            return Status::OK();
          }
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
          return status;
        } else {
          LOG(INFO) << "Something bad happened in fastq reader: "
            << status.ToString() << std::endl;
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
      Env* const env_;
      int64 line_number_ = 0;
      std::unique_ptr<io::InputBuffer> input_buffer_;
  };

  class FastqReaderOp : public ReaderOpKernel {
    public:
      explicit FastqReaderOp(OpKernelConstruction* context)
        : ReaderOpKernel(context) {

          Env* env = context->env();
          SetReaderFactory([this, env]() {
              return new FastqReader(name(), env);
              });
        }
  };

  REGISTER_KERNEL_BUILDER(Name("FastqReader").Device(DEVICE_CPU),
      FastqReaderOp);

}  // namespace tensorflow
