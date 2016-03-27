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

// Reader OP to write SAM files and returns 
#include <memory>
#include <vector>
#include "tensorflow/core/framework/writer_op_kernel.h"
#include "tensorflow/core/kernels/writer_base.h"
#include "tensorflow/core/user_ops/dna-align/snap_proto.pb.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/Read.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/AlignmentResult.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("SamWriter")
    .Output("writer_handle: Ref(string)")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("out_file: string")
    .SetIsStateful()
    .Doc(R"doc(
A writer that writes aligned reads to a SAM file.

writer_handle: The handle to reference the Writer.
container: If non-empty, this writer is placed in the given container.
        Otherwise, a default container is used.
)doc");

class SamWriter : public WriterBase {
 public:
     // TODO(stuart) modify to pass references to aligneroptions and genome
     // resources
  SamWriter(const string& node_name, Env* env, const string& work)
      : WriterBase(strings::StrCat("SamWriter '", node_name, "'"), work),
        env_(env) {}

  Status OnWorkStartedLocked() override {
    record_number_ = 0;
    TF_RETURN_IF_ERROR(env_->NewWritableFile(current_work(), &out_file_));
   
    // write SAM file header (need AlignerOptions, genome)
    return Status::OK();
  }

  Status OnWorkFinishedLocked() override {
    out_file_->Close();
    return Status::OK();
  }

  Status WriteLocked(string* value) override {

    //LOG(INFO) << "Reading from file " << current_work() << " !\n";

    // `value` is a serialized AlignmentDef protobuf message
    // get submessage ReadDef, write each SingleResult to file

    SnapProto::AlignmentDef alignment;
    if (!alignment.ParseFromString(*value)) {
        return errors::Internal("Failed to parse AlignmentDef",
                " from string in SamWriter WriteLocked()");
    }

    SnapProto::ReadDef& read = alignment.read();
    Read snap_read;
    snap_read.init(
        read_proto.meta().c_str(),
        read_proto.meta().length(),
        read_proto.bases().c_str(),
        read_proto.qualities().c_str(),
        read_proto.length()
    );

    SingleAlignmentResult* snap_results = new SingleAlignmentResult[alignment.results_size()];
    
    LOG(INFO) << "Preparing " << results_proto.results_size() << " results for writing to file."
              << " for read " << read_proto.meta() << "  " << read_proto.bases();
    LOG(INFO) << "firstIsPrimary is " << results_proto.firstIsPrimary() ? "true" : "false";

    for (int j = 0; j < alignment.results_size(); j++) {
        const SnapProto::SingleResult& single_result = alignment.results(j);
        snap_results[j].status = (AlignmentResult)single_result.result();
        snap_results[j].genomeLocation = GenomeLocation(single_result.genomeLocation());
        snap_results[j].direction = (Direction)single_result.direction();
        snap_results[j].mapq = single_result.mapq();
    }
        
    //read_writer_->writeReads(reader_context_, &snap_read, snap_results, results_proto.results_size(), results_proto.firstIsPrimary()); 
    delete [] snap_results;
  }

 private:
  WritableFile* out_file_ = nullptr;
  Env* const env_;
  int64 record_number_;
};

class SamWriterOp : public WriterOpKernel {
 public:
  explicit SamWriterOp(OpKernelConstruction* context)
      : WriterOpKernel(context) {
    string out_file;
    OP_REQUIRES_OK(context,
                   context->GetAttr("out_file", &out_file));
    Env* env = context->env();
    SetWriterFactory([this, out_file, env]() {
      return new SamWriter(name(), env, out_file);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("SamWriter").Device(DEVICE_CPU),
                        SamWriterOp);

}  // namespace tensorflow
