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
#include "tensorflow/core/user_ops/dna-align/snap_read_decode.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("FastqReader")
  .Output("reader_handle: Ref(string)")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("batch_size: int")
  .SetIsStateful()
  .Doc(R"doc(
  A Reader that outputs the read sequences in a FASTQ file. 

  reader_handle: The handle to reference the Reader.
  skip_header_lines: Number of lines to skip from the beginning of every file.
  container: If non-empty, this reader is placed in the given container.
  Otherwise, a default container is used.
  )doc");

class FastqReader : public ReaderBase {
  public:
    FastqReader(const string& node_name, Env* env, int batch_size)
      : ReaderBase(strings::StrCat("FastqReader '", node_name, "'")),
      env_(env),
      batch_size_(batch_size),
      line_number_(0),
      num_produced_(0){}

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
      
      VLOG(INFO) << "Opening file: " << current_work() << std::endl;
      return Status::OK();
    }

    Status OnWorkFinishedLocked() override {
      return Status::OK();
    }
    
    TensorShape GetUserRequiredShape() override {
      return TensorShape({3, batch_size_});
    }

    DataType GetUserRequiredType() override {
      return DT_STRING;
    }

    Status ReadBatchLocked(Tensor* tensor_batch, 
        int* num_produced, bool* at_end) override {

      string lines[4];
      Status status;
      MutableSnapReadDecode reads(tensor_batch);
      for (int j = 0; j < reads.size(); j++) {

        for (int i = 0; i < 4; i++)
        {
          status = ReadLine_(&lines[i]);
          ++line_number_;
          if (errors::IsOutOfRange(status)) {
            LOG(INFO) << "I is out of range! file: " 
              << current_work() << " line number: " << line_number_ << std::endl;
            *at_end = true;
            // fill the rest with blanks, partial batch
            LOG(INFO) << "filling the rest (" << reads.size()-j << ")  with blanks!";
            string blank = "";
            for (int k = j; k < reads.size(); k++)
              reads.set_bases(k, blank);
            // caller will move to next file, everything still OK
            return Status::OK();
          }
        }

        if (status.ok()) {
          lines[0].erase(0, 1);  // remove the '@' from meta data
          reads.set_bases(j, lines[1]);
          reads.set_metadata(j, lines[0]);
          reads.set_qualities(j, lines[3]);
          (*num_produced)++;
          num_produced_++;
          //LOG(INFO) << "num produced is now: " << num_produced_;
        } else {
          LOG(INFO) << "Something bad happened in fastq reader read batch: "
            << status.ToString();
          return status;
        }
      }
      return Status::OK();
    }
      

    Status ReadLocked(string* key, string* value, bool* produced,
        bool* at_end) override {

      //LOG(INFO) << "Reading from file " << current_work() << " !\n";

      string lines[4];
      Status status;
      for (int i = 0; i < 4; i++)
      {
        status = ReadLine_(&lines[i]);
        ++line_number_;
        if (errors::IsOutOfRange(status)) {
          LOG(INFO) << "I is out of range! file: " 
            << current_work() << " line number: " << line_number_ << std::endl;
          *at_end = true;
          return Status::OK();
        }
      }

      if (status.ok()) {
        lines[0].erase(0, 1);  // remove the '@' from meta data
        SnapProto::AlignmentDef alignment;
        SnapProto::ReadDef* read = alignment.mutable_read();
        read->set_bases(lines[1]);
        read->set_meta(lines[0]);
        read->set_length(lines[1].length());
        read->set_qualities(lines[3]);
        alignment.SerializeToString(value);
        *key = strings::StrCat(current_work(), ":", line_number_);
        *produced = true;
        num_produced_++;
        return status;
      } else {
        LOG(INFO) << "Something bad happened in fastq reader: "
          << status.ToString() << std::endl;
        return status;
      }

    }

    Status ResetLocked() override {
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
    Env* const env_;
    int batch_size_;
    std::shared_ptr<ReadOnlyMemoryRegion> mmap_fastq_;
    const char* mmap_data_;
    uint64 bytes_;
    int64 line_number_ = 0;
    int64 num_produced_ = 0;
};

class FastqReaderOp : public ReaderOpKernel {
  public:
    explicit FastqReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
 
        int batch_size;
        OP_REQUIRES_OK(context, context->GetAttr("batch_size", 
              &batch_size));
        Env* env = context->env();
        SetReaderFactory([this, env, batch_size]() {
            return new FastqReader(name(), env, batch_size);
            });
      }
};

REGISTER_KERNEL_BUILDER(Name("FastqReader").Device(DEVICE_CPU),
    FastqReaderOp);

}  // namespace tensorflow
