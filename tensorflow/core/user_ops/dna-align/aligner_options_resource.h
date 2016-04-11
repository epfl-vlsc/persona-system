
#ifndef TENSORFLOW_FRAMEWORK_ALIGNEROPTIONSRESOURCE_H_
#define TENSORFLOW_FRAMEWORK_ALIGNEROPTIONSRESOURCE_H_

#include <memory>
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/AlignerOptions.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include <ctype.h>

namespace tensorflow {

    class AlignerOptionsResource : public ResourceBase {
    public:
        explicit AlignerOptionsResource() {}

        AlignerOptions* value() { return value_; }

        void init(string cmdLine) {
            value_ = new AlignerOptions("AlignerOptions!!");
            // may need to trim cmdLine?
            strcpy(cmd_line_, cmdLine.c_str());
            char* pChar = cmd_line_;
            int argc = 1; // we assume there'll always be something
            while (*pChar) {
                if (isspace(*pChar)) {
                    argc++;
                    while (isspace(*pChar)) {
                        *pChar = '\0';
                        pChar++; // multi-space
                    }
                }
                pChar++;
            }
            char** argv = new char*[argc];
            pChar = cmd_line_;
            argv[0] = cmd_line_;
            int count = 1;
            while (count < argc) {
                if (*pChar == '\0') {
                    while (*pChar == '\0') pChar++;
                    argv[count] = pChar;
                    count++;
                }
                pChar++;
            }

            bool ignored = false;
            // SNAP's options parsing is weird, it only parses 1 arg at a time
            for (int n = 0; n < argc; ++n) {
                bool success = value_->parse((const char**)argv, argc, n, &ignored);

                if (!success) {
                    LOG(INFO) << "ERROR: AlignerOptionsResource::parse failed at index " << n;
                    break;
                }
            }

            delete[] argv;
            //LOG(INFO) << "useM is = " << value_->useM;
        }

        string DebugString() override {
            return "SNAP Aligneroptions";
        }

    private:
        AlignerOptions* value_;
        char cmd_line_[512]; // safe enough length?

    };

} // namespace tensorflow

#endif
