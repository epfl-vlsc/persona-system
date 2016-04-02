
#ifndef TENSORFLOW_FRAMEWORK_GENOMEINDEXRESOURCE_H_
#define TENSORFLOW_FRAMEWORK_GENOMEINDEXRESOURCE_H_

#include <memory>
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/GenomeIndex.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
  
class GenomeIndexResource : public ResourceBase {
    public:
        explicit GenomeIndexResource() {}

        GenomeIndex* get_index() { return value_; }
        const Genome* get_genome() { return value_->getGenome(); }

        void init(string path) {
            // 2nd and 3rd arguments are weird SNAP things that can safely be ignored
            value_ = GenomeIndex::loadFromDirectory(const_cast<char*>(path.c_str()), false, false);
        }

        string DebugString() override {
            return "SNAP GenomeIndex";
        }

    private:
        GenomeIndex* value_;

        TF_DISALLOW_COPY_AND_ASSIGN(GenomeIndexResource);
};

} // namespace tensorflow

#endif
