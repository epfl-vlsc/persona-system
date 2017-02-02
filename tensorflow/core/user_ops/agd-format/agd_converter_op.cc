#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "read_resource.h"
#include "format.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;

  namespace {
    const string op_name("AGDConverter");
  }

  REGISTER_OP(op_name.c_str())
  .Input("buffer_list_pool: Ref(string)")
  .Input("input_data: string")
  .Output("agd_columns: string")
  .SetShapeFn([](shape_inference::InferenceContext *c) {
      using namespace shape_inference;

      ShapeHandle input_data;
      for (int i = 0; i < 2; i++) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));
        auto dim_handle = c->Dim(input_data, 0);
        auto dim_value = c->Value(dim_handle);
        if (dim_value != 2) {
          return Internal("AGDConverter input ", i, " must be a vector(2)");
        }
      }
      c->set_output(0, input_data);

      return Status::OK();
    })
  .Doc(R"doc(
Converts an input file into three files of bases, qualities, and metadata
)doc");

  class AGDConverterOp : public OpKernel {
  public:
    AGDConverterOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    }

    ~AGDConverterOp() {
      core::ScopedUnref a(buflist_pool_);
    }

    void Compute(OpKernelContext* ctx) override {
      if (!buflist_pool_) {
        OP_REQUIRES_OK(ctx, Init(ctx));
      }

      ResourceContainer<ReadResource> *reads_container;
      ResourceContainer<BufferList> *buffer_list_resource;
      OP_REQUIRES_OK(ctx, GetResources(ctx, &reads_container, &buffer_list_resource));

      auto *read_resource = reads_container->get();
      auto *buffer_list = buffer_list_resource->get();

      // These nested scopes are to make sure the RAII fires off in-order
      core::ScopedUnref a1(reads_container);
      {
        ResourceReleaser<ReadResource> a2(*reads_container);
        {
          ReadResourceReleaser a3(*read_resource);
          buffer_list->resize(3);

          auto &bases = (*buffer_list)[0];
          auto &qualities = (*buffer_list)[1];
          auto &metadata = (*buffer_list)[2];

          const char *read_data;
          unsigned read_data_length;
          Read read;
          auto status = read_resource->get_next_record(read);
          while (status.ok()) {

            OP_REQUIRES_OK(ctx, AppendRecord(read.getId(), read.getIdLength(), metadata));
            read_data_length = read.getDataLength();
            OP_REQUIRES_OK(ctx, AppendRecord(read.getQuality(), read_data_length, qualities));
            OP_REQUIRES_OK(ctx, IntoBases(read.getData(), read_data_length, bases_));
            OP_REQUIRES_OK(ctx, AppendRecord(&bases_[0], bases_.size(), bases));

            status = read_resource->get_next_record(read);
          }

          OP_REQUIRES(ctx, IsResourceExhausted(status), status);
        }
      }
    }

  private:
    ReferencePool<BufferList> *buflist_pool_ = nullptr;
    vector<BinaryBases> bases_;

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
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &buflist_pool_));
      return Status::OK();
    }

    Status GetResources(OpKernelContext *ctx,
                        ResourceContainer<ReadResource> **reads_container,
                        ResourceContainer<BufferList> **buffer_list) {
      const Tensor *input_data_t;
      TF_RETURN_IF_ERROR(ctx->input("input_data", &input_data_t));
      auto input_data_v = input_data_t->vec<string>();
      TF_RETURN_IF_ERROR(ctx->resource_manager()->Lookup(input_data_v(0), input_data_v(1),
                                                         reads_container));
      TF_RETURN_IF_ERROR(buflist_pool_->GetResource(buffer_list));
      (*buffer_list)->get()->reset();
      TF_RETURN_IF_ERROR((*buffer_list)->allocate_output("agd_columns", ctx));
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name(op_name.c_str()).Device(DEVICE_CPU), AGDConverterOp);
} // namespace tensorflow {
