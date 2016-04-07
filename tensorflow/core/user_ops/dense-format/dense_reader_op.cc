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

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/kernels/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/user_ops/format.h"

namespace tensorflow {
  
class DenseReader : public ReaderBase {
public:
  DenseReader(const string& node_name, Env* env)
    : ReaderBase(strings::StrCat("DenseReader '", node_name, "'")),
      env_(env) {};

  Status OnWorkStartedLocked() override {
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
