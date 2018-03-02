
// Stuart Byma
// Op providing SNAP genome index and genome

#include <unistd.h>
#include <memory>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/contrib/persona/kernels/agd-ops/agd_reference_genome.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

    class AGDReferenceGenomeOp : public OpKernel {
    public:
      typedef BasicContainer<AGDReferenceGenome> GenomeContainer;

        AGDReferenceGenomeOp(OpKernelConstruction* context)
            : OpKernel(context), genome_handle_set_(false) {
          //OP_REQUIRES_OK(context, context->GetAttr("genome_location", &genome_location_));
          
          OP_REQUIRES_OK(context, context->GetAttr("chunk_paths", &chunk_paths_));

          //OP_REQUIRES_OK(context, context->env()->FileExists(genome_location_));
          OP_REQUIRES_OK(context,
                         context->allocate_persistent(DT_STRING, TensorShape({ 2 }),
                                                      &genome_handle_, nullptr));
        }

        void Compute(OpKernelContext* ctx) override {
            mutex_lock l(mu_);
            if (!genome_handle_set_) {
                OP_REQUIRES_OK(ctx, SetGenomeHandle(ctx, chunk_paths_));
            }
            ctx->set_output_ref(0, &mu_, genome_handle_.AccessTensor(ctx));
        }

    protected:
        ~AGDReferenceGenomeOp() override {
            // If the genome object was not shared, delete it.
            if (genome_handle_set_ && cinfo_.resource_is_private_to_kernel()) {
                TF_CHECK_OK(cinfo_.resource_manager()->Delete<GenomeContainer>(
                    cinfo_.container(), cinfo_.name()));
            }
        }

    protected:

        ContainerInfo cinfo_;

    private:
        Status SetGenomeHandle(OpKernelContext* ctx, vector<string>& chunk_paths) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
            TF_RETURN_IF_ERROR(cinfo_.Init(ctx->resource_manager(), def()));
            GenomeContainer* reference_genome;

            auto creator = [this, chunk_paths, ctx](GenomeContainer** genome) {
                LOG(INFO) << "loading AGD reference genome ...";
                auto begin = std::chrono::high_resolution_clock::now();
                // map files
                vector<const char*> base_chunks, meta_chunks;
                vector<uint32_t> base_lens, meta_lens;
                vector<std::unique_ptr<ReadOnlyMemoryRegion>> base_mmaps;
                vector<std::unique_ptr<ReadOnlyMemoryRegion>> meta_mmaps;
                base_mmaps.resize(chunk_paths.size()); meta_mmaps.resize(chunk_paths.size());

                for (int i = 0; i < chunk_paths.size(); i++) {

                  LOG(INFO) << "ref genome op loading chunk file: " << (chunk_paths[i] + ".base");
                  TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile( 
                        chunk_paths[i] + ".base", &base_mmaps[i]));
                  base_chunks.push_back((const char *)base_mmaps[i]->data());
                  base_lens.push_back(base_mmaps[i]->length());
                  LOG(INFO) << "ref genome op loading chunk file: " << (chunk_paths[i] + ".meta");
                  TF_RETURN_IF_ERROR(ctx->env()->NewReadOnlyMemoryRegionFromFile( 
                        chunk_paths[i] + ".meta", &meta_mmaps[i]));
                  meta_chunks.push_back((const char *)meta_mmaps[i]->data());
                  meta_lens.push_back(meta_mmaps[i]->length());
                }

                // init ref genome object
                AGDReferenceGenome* g = new AGDReferenceGenome();
                Status s = AGDReferenceGenome::Create(g, base_chunks, base_lens, meta_chunks, meta_lens);
                if (!s.ok()) {
                  delete g;
                  return s;
                }
                unique_ptr<AGDReferenceGenome> value(g);

                // unmap files, happens on vec destruction

                auto end = std::chrono::high_resolution_clock::now();
                LOG(INFO) << "genome load time is: " << ((float)std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count())/1000000000.0f;
                *genome = new GenomeContainer(move(value));
                return Status::OK();
            };

            TF_RETURN_IF_ERROR(
                cinfo_.resource_manager()->LookupOrCreate<GenomeContainer>(
                    cinfo_.container(), cinfo_.name(), &reference_genome, creator));

            auto h = genome_handle_.AccessTensor(ctx)->flat<string>();
            h(0) = cinfo_.container();
            h(1) = cinfo_.name();
            genome_handle_set_ = true;
            return Status::OK();
        }

        mutex mu_;
        //string genome_location_;
        PersistentTensor genome_handle_ GUARDED_BY(mu_);
        bool genome_handle_set_ GUARDED_BY(mu_);
        vector<string> chunk_paths_;
    };

    REGISTER_KERNEL_BUILDER(Name("GenomeIndex").Device(DEVICE_CPU), AGDReferenceGenomeOp);
}  // namespace tensorflow
