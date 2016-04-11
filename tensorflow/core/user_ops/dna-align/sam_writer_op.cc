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
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/kernels/writer_base.h"
#include "tensorflow/core/user_ops/dna-align/snap_proto.pb.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/Read.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/AlignmentResult.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/AlignerOptions.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/FileFormat.h"
#include "tensorflow/core/user_ops/dna-align/aligner_options_resource.h"
#include "tensorflow/core/user_ops/dna-align/genome_index_resource.h"
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
        /*    .Attr("chromosome_padding: int")
            .Attr("genome_container: string = ''")
            .Attr("genome_shared_name: string = ''")
            .Attr("options_container: string = ''")
            .Attr("options_shared_name: string = ''")*/
        .Attr("out_file: string")
        .SetIsStateful()
        .Doc(R"doc(
A writer that writes aligned reads to stdout to test.

writer_handle: The handle to reference the Writer.
out_file: The file to write results to.
)doc");

    class SamWriter : public WriterBase {
    public:

        SamWriter(const string& node_name, Env* env, const string& work)
            : WriterBase(strings::StrCat("SamWriter'", node_name, "'"), work),
            env_(env) {}

        ~SamWriter() {
            delete aligner_options_;
        }

        Status OnWorkStartedLocked(OpKernelContext* context) override {
            record_number_ = 0;
            if (!genome_index_) {

                string options_container;
                string genome_container;
                string options_shared_name;
                string genome_shared_name;

                OpMutableInputList meta_list;
                Status status = context->mutable_input_list("meta_handles", &meta_list);
                if (!status.ok()) {
                    LOG(INFO) << "Failed to get meta handles! " << status.ToString();
                }

                Tensor options_handle = meta_list.at(0, false);
                if (options_handle.NumElements() != 2) {
                    return errors::InvalidArgument(
                        "Metadata handle for SamWriter must have 2 elements ",
                        options_handle.shape().DebugString());
                }
                Tensor genome_handle = meta_list.at(1, false);
                if (genome_handle.NumElements() != 2) {
                    return errors::InvalidArgument(
                        "Metadata genome handle for SamWriter must have 2 elements ",
                        genome_handle.shape().DebugString());
                }
                options_container = options_handle.flat<string>()(0);
                options_shared_name = options_handle.flat<string>()(1);
                genome_container = genome_handle.flat<string>()(0);
                genome_shared_name = genome_handle.flat<string>()(1);
                /*LOG(INFO) << options_container << " | " << options_shared_name
                    << " | " << genome_container << " | " << genome_shared_name;*/

                AlignerOptionsResource* options;
                status = context->resource_manager()->Lookup(options_container,
                    options_shared_name, &options);
                if (!TF_PREDICT_TRUE(status.ok())) {
                    LOG(INFO) << "failed to get aligner options: " << status.ToString();
                    context->CtxFailure(status);
                    return status;
                }
                status = context->resource_manager()->Lookup(genome_container,
                    genome_shared_name, &genome_index_);
                if (!TF_PREDICT_TRUE(status.ok())) {
                    LOG(INFO) << "failed to get genome: " << status.ToString();
                    context->CtxFailure(status);
                    return status;
                }
                // hack -- copy aligner options so we can assign a different filename
                // for each writer. Alternative is changing SNAP which we prefer to 
                // avoid
                aligner_options_ = new AlignerOptions("Copy of the options!");
                *aligner_options_ = *(options->value());
                aligner_options_->outputFile.fileName = current_work().c_str();
                options->Unref();

                argv_ = new char*();
                argv_[0] = new char[128]();
                strcpy(argv_[0], "TFBioFLow!");
            }
            //TF_RETURN_IF_ERROR(env_->NewWritableFile(current_work(), &out_file_));
            //LOG(INFO) << "Opening file " << current_work(); 

            memset(&reader_context_, 0, sizeof(reader_context_));
            reader_context_.clipping = aligner_options_->clipping;
            reader_context_.defaultReadGroup = aligner_options_->defaultReadGroup;
            reader_context_.genome = genome_index_->get_genome();
            reader_context_.ignoreSecondaryAlignments = aligner_options_->ignoreSecondaryAlignments;
            reader_context_.ignoreSupplementaryAlignments = aligner_options_->ignoreSecondaryAlignments;
            DataSupplier::ExpansionFactor = aligner_options_->expansionFactor;

            const FileFormat* format;
            if (aligner_options_->useM)
              format = FileFormat::SAM[true];
            else
              format = FileFormat::SAM[false];

            format->setupReaderContext(aligner_options_, &reader_context_);

            writer_supplier_ = format->getWriterSupplier(aligner_options_, reader_context_.genome);
            read_writer_ = writer_supplier_->getWriter();
            read_writer_->writeHeader(reader_context_, aligner_options_->sortOutput, 1, (const char**)argv_,
                "version", aligner_options_->rgLineContents, aligner_options_->outputFile.omitSQLines);

            return Status::OK();
        }

        Status OnWorkFinishedLocked() override {
            //out_file_->Close();
            read_writer_->close();
            writer_supplier_->close();
            return Status::OK();
        }

        Status WriteLocked(const string& value) override {
            if (!genome_index_) {
                LOG(INFO) << "WTF genome index is null!";
            }
            LOG(INFO) << "write is being called!";
            // `value` is a serialized AlignmentDef protobuf message
            // get submessage ReadDef, write each SingleResult to file

            /*SnapProto::AlignmentDef alignment;
            if (!alignment.ParseFromString(value)) {
                return errors::Internal("Failed to parse AlignmentDef",
                    " from string in SamWriter WriteLocked()");
            }

            const SnapProto::ReadDef* read = &alignment.read();
            Read snap_read;
            snap_read.init(
                read->meta().c_str(),
                read->meta().length(),
                read->bases().c_str(),
                read->qualities().c_str(),
                read->length()
            );

            SingleAlignmentResult* snap_results;
            if (alignment.results_size() > 0) {
                snap_results = new SingleAlignmentResult[alignment.results_size()];
            }
            else {
                return errors::Internal("SamWriter Error: Alignment had no results!");
            }

            // debugging
            LOG(INFO) << "Preparing " << alignment.results_size() << " results for writing to file."
                << " for read " << read->meta() << "  " << read->bases();
            string isp = alignment.firstisprimary() ? "true" : "false";
            LOG(INFO) << "firstIsPrimary is " << isp;*/

            /*if (alignment.results_size() == 0) {
                LOG(INFO) << "There were 0 results in this read";
            }

            for (int j = 0; j < alignment.results_size(); j++) {
                const SnapProto::SingleResultDef& single_result = alignment.results(j);
                snap_results[j].status = (AlignmentResult)single_result.result();
                snap_results[j].location = GenomeLocation(single_result.genomelocation());
                snap_results[j].direction = (Direction)single_result.direction();
                snap_results[j].mapq = single_result.mapq();
                LOG(INFO) << " result: location " << snap_results[j].location <<
                    " direction: " << snap_results[j].direction << " score " << snap_results[j].score;
            }

            record_number_++;
            read_writer_->writeReads(reader_context_, &snap_read, snap_results, alignment.results_size(), alignment.firstisprimary());
            delete[] snap_results;*/
            return Status::OK();
        }

    private:
        //WritableFile* out_file_ = nullptr;
        Env* const env_;
        int64 record_number_;
        GenomeIndexResource* genome_index_ = nullptr;
        AlignerOptions* aligner_options_ = nullptr;
        ReaderContext reader_context_;
        ReadWriterSupplier *writer_supplier_;
        ReadWriter* read_writer_;
        char** argv_;

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
