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
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

    REGISTER_OP("FastqReader")
        .Output("reader_handle: Ref(string)")
        .Attr("container: string = ''")
        .Attr("shared_name: string = ''")
        .Attr("add_record_name: bool = false")
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
      FastqReader(const string& node_name, Env* env, const bool add_record_name)
            : ReaderBase(strings::StrCat("FastqReader '", node_name, "'")),
            env_(env),
            line_number_(0),
              num_produced_(0),
              add_record_name_(add_record_name) {}

        Status OnWorkStartedLocked() override {
            line_number_ = 0;
            num_produced_ = 0;
            ReadOnlyMemoryRegion *raw_mmap = nullptr;
            Status status = env_->NewReadOnlyMemoryRegionFromFile(current_work(),
                &raw_mmap);
            if (!status.ok()) {
                LOG(INFO) << "ERROR: problem creating mmap file in fastqreader";
                return status;
            }

            mmap_fastq_ = std::shared_ptr<ReadOnlyMemoryRegion>(raw_mmap);

            mmap_data_ = reinterpret_cast<const char*>(mmap_fastq_->data());
            bytes_ = 0;

            LOG(INFO) << "Opening file: " << current_work() << std::endl;
            return Status::OK();
        }

        Status OnWorkFinishedLocked() override {
            return Status::OK();
        }

        Status ReadLocked(string* key, string* value, bool* produced,
            bool* at_end) override {

            LOG(INFO) << "Reading from file " << current_work() << " !\n";

            string lines[4];
            Status status;
            for (int i = 0; i < 4; i++)
            {
                LOG(INFO) << "Reading i = " << i;
                status = ReadLine_(&lines[i]);
                ++line_number_;
                if (errors::IsOutOfRange(status)) {
                    LOG(INFO) << "Out of range! line number: " << line_number_;
                    *at_end = true;
                    return Status::OK();
                }
                LOG(INFO) << "End reading i = " << i;
            }

            if (status.ok()) {
                LOG(INFO) << "Doing stuff";

                lines[0].erase(0, 1);  // remove the '@' from meta data
                SnapProto::AlignmentDef alignment;
                SnapProto::ReadDef* read = alignment.mutable_read();
                read->set_bases(lines[1]);
                read->set_meta(lines[0]);
                read->set_length(lines[1].length());
                read->set_qualities(lines[3]);
                if (add_record_name_) {
                  read->set_record_name(current_work());
                }

                LOG(INFO) << "Serializing alignment";

                alignment.SerializeToString(value);
                *key = strings::StrCat(current_work(), ":", line_number_);
                *produced = true;
                num_produced_++;

                LOG(INFO) << "End of a read.";

                return status;
            }
            else {
                LOG(INFO) << "Something bad happened in fastq reader: "
                    << status.ToString() << std::endl;
                return status;
            }

        }

        Status ResetLocked() override {
            LOG(INFO) << "Resetting";

            line_number_ = 0;
            num_produced_ = 0;
            return ReaderBase::ResetLocked();
        }

    private:

        Status ReadLine_(string* result) {
            if (bytes_ >= mmap_fastq_->length())
                return errors::OutOfRange("eof");
            const char* orig = mmap_data_;
            while (*mmap_data_ != '\n' && bytes_ < mmap_fastq_->length()) {
                // EOF is treated like end of line
                bytes_++;
                mmap_data_++;
            }
            if (mmap_data_ == orig)
                result->assign("\n");
            else
                result->assign(orig, (size_t)(mmap_data_ - orig));
            // advance past the newline character
            mmap_data_++;
            bytes_++;
            return Status::OK();
        }
        enum { kBufferSize = 256 << 10 /* 256 kB */ };
        Env* const env_;
        std::shared_ptr<ReadOnlyMemoryRegion> mmap_fastq_;
        const char* mmap_data_;
        uint64 bytes_;
        int64 line_number_ = 0;
        int64 num_produced_ = 0;
        bool add_record_name_ = false;
    };

    class FastqReaderOp : public ReaderOpKernel {
    public:
        explicit FastqReaderOp(OpKernelConstruction* context)
            : ReaderOpKernel(context) {
            bool add_record_name = false;

            Env* env = context->env();
            OP_REQUIRES_OK(context,
                           context->GetAttr("add_record_name", &add_record_name));
            SetReaderFactory([this, env, add_record_name]() {
                return new FastqReader(name(), env, add_record_name);
            });
        }
    };

    REGISTER_KERNEL_BUILDER(Name("FastqReader").Device(DEVICE_CPU),
        FastqReaderOp);

}  // namespace tensorflow
