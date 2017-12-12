#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/agd-format/read_resource.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;

  namespace {
    const string op_name("AGDFastaConverter");
  }


  class AGDFastaConverterOp : public OpKernel {
  public:
    AGDFastaConverterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("is_nucleotide", &is_nucleotide_));
    }

    ~AGDFastaConverterOp() {
      core::ScopedUnref a(bufpair_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!bufpair_pool_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      ResourceContainer<ReadResource> *reads_container;
      ResourceContainer<BufferPair> *bases_container, *meta_container;
      OP_REQUIRES_OK(ctx, GetResources(ctx, &reads_container, &bases_container,
      &meta_container));

      auto *read_resource = reads_container->get();

      // These nested scopes are to make sure the RAII fires off in-order
      core::ScopedUnref a1(reads_container);
      {
        ResourceReleaser<ReadResource> a2(*reads_container);
        {
          ReadResourceReleaser a3(*read_resource);
          auto& bases_buf = *bases_container->get();
          auto& metadata_buf = *meta_container->get();


          size_t meta_len;
          size_t bases_len;
          const char* bases, *qual, *meta;
          auto status = read_resource->get_next_record(&bases, &bases_len, &qual, &meta, &meta_len);
          if (qual != nullptr)
            LOG(INFO) << "FASTA converter got non-null qual data, are you sure this is FASTA data?";

          while (status.ok()) {

            OP_REQUIRES_OK(ctx, AppendRecord(meta, meta_len, metadata_buf));
            //LOG(INFO) << "0: meta: " << string(meta, meta_len);
            //LOG(INFO) << "0: base: " << string(bases, bases_len);
            OP_REQUIRES_OK(ctx, IntoBases(bases, bases_len, bases_));
            OP_REQUIRES_OK(ctx, AppendRecord(&bases_[0], bases_.size(), bases_buf));

            status = read_resource->get_next_record(&bases, &bases_len, &qual, &meta, &meta_len);
          }

          OP_REQUIRES(ctx, IsResourceExhausted(status), status);
        }
      }
    }

  private:
    ReferencePool<BufferPair> *bufpair_pool_ = nullptr;
    vector<BinaryBases> bases_;
    bool is_nucleotide_ = true;

    template <typename T>
    Status AppendRecord(const T* data, unsigned elements_t, BufferPair &bp) {
      auto &index = bp.index();
      auto &data_buf = bp.data();

      int64_t elements = elements_t * sizeof(T);
      if (elements > MAX_INDEX_SIZE) {
        return Internal("Record size in bytes (", elements, ") exceeds the maximum (", MAX_INDEX_SIZE, ")");
      }

      auto converted_size = static_cast<RelativeIndex>(elements);
      auto converted_data = reinterpret_cast<const char*>(data);

      TF_RETURN_IF_ERROR(index.AppendBuffer(reinterpret_cast<char*>(&converted_size), sizeof(converted_size)));
      TF_RETURN_IF_ERROR(data_buf.AppendBuffer(converted_data, converted_size));

      return Status::OK();
    }

    Status Init(OpKernelContext* ctx) {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pair_pool", &bufpair_pool_));
      return Status::OK();
    }

    Status GetResources(OpKernelContext *ctx,
                        ResourceContainer<ReadResource> **reads_container,
                        ResourceContainer<BufferPair> **base_bufpair,
                        ResourceContainer<BufferPair> **meta_bufpair) {
      const Tensor *input_data_t;
      TF_RETURN_IF_ERROR(ctx->input("input_data", &input_data_t));
      auto input_data_v = input_data_t->vec<string>();
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(input_data_v(0), input_data_v(1),
                                                         reads_container));
      
      TF_RETURN_IF_ERROR(bufpair_pool_->GetResource(base_bufpair));
      TF_RETURN_IF_ERROR(bufpair_pool_->GetResource(meta_bufpair));
      (*base_bufpair)->get()->reset();
      (*meta_bufpair)->get()->reset();
      TF_RETURN_IF_ERROR((*base_bufpair)->allocate_output("bases_out", ctx));
      TF_RETURN_IF_ERROR((*meta_bufpair)->allocate_output("meta_out", ctx));
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDFastaConverterOp);
} // namespace tensorflow {
