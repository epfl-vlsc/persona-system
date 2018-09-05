
#include "tensorflow/contrib/persona/kernels/protein-cluster/candidate_map.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

namespace tensorflow {

// Defines a QueueOp, an abstract class for Queue construction ops.
class CandidateMapOp : public ResourceOpKernel<CandidateMap> {
 public:
  CandidateMapOp(OpKernelConstruction* context) : ResourceOpKernel(context) {}

 private:
  Status CreateResource(CandidateMap** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    CandidateMap* map = new CandidateMap();
    *ret = map;
    return Status::OK();
  }
  
  TF_DISALLOW_COPY_AND_ASSIGN(CandidateMapOp);

};

REGISTER_KERNEL_BUILDER(Name("CandidateMap").Device(DEVICE_CPU), CandidateMapOp);

}
