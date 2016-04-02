
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
            int argc = 0;
            while (*pChar) {
                if (isspace(*pChar)) {
                    argc++;
                    while(isspace(*pChar)) {
                        *pChar = '\0';
                        pChar++; // multi-space
                    }
                }
                pChar++;
            }
            char** argv = new char*[argc];
            pChar = cmd_line_;
            pChar++;
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
            bool done; 
            // a '2' is here because the first two cmd line options
            // are the program and single or paired. The SNAP 'parse'
            // function skips these. 
            int start = 2;
            bool success = value_->parse((const char**)argv, argc, start, &done);
            if (!success)
                LOG(INFO) << "ERROR: AlignerOptionsResource::parse failed!";
            delete [] argv;
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
