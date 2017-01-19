#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "format.h"
#include "column_builder.h"
#include "agd_result_reader.h"
#include "compression.h"
#include "util.h"
#include "buffer.h"
#include <vector>
#include <cstdint>
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "tensorflow/core/user_ops/object-pool/ref_pool.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"
#include <boost/functional/hash.hpp>
#include <google/dense_hash_map>

namespace tensorflow {

   namespace { 
      void resource_releaser(ResourceContainer<Data> *data) {
        core::ScopedUnref a(data);
        data->release();
      }
   }

  REGISTER_OP("AGDMarkDuplicates")
  .Input("buffer_list_pool: Ref(string)")
  .Input("results_handle: Ref(string)")
  .Input("num_records: int32")
  .Output("marked_results: string")
  .SetIsStateful()
  .Doc(R"doc(
Mark duplicate reads/pairs that map to the same location. 

This Op depends on data being sorted by metadata (QNAME), 
i.e. A paired read is immediately followed by its mate. 

Normally this step would be run on the aligner output before
sorting by genome location.

The implementation follows the approach used by SamBlaster
github.com/GregoryFaust/samblaster
wherein read pair signatures are looked up in a hash table
to determine if there are reads/pairs mapped to the exact 
same location. Our implementation uses google::dense_hash_table,
trading memory for faster execution. 
  )doc");

  using namespace std;
  using namespace errors;
  using namespace format;

  class AGDMarkDuplicateOp : public OpKernel {
  public:
    AGDMarkDuplicateOp(OpKernelConstruction *context) : OpKernel(context) {
      signature_map_ = new SignatureMap();
      signature_map_->set_deleted_key(Signature());
    }

    ~AGDMarkDuplicateOp() {
      core::ScopedUnref unref_listpool(bufferlist_pool_);
      delete signature_map_;
    }

    Status GetOutputBufferList(OpKernelContext* ctx, ResourceContainer<BufferList> **ctr)
    {
      TF_RETURN_IF_ERROR(bufferlist_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      TF_RETURN_IF_ERROR((*ctr)->allocate_output("marked_results", ctx));
      return Status::OK();
    }
    
    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_list_pool", &bufferlist_pool_));

      return Status::OK();
    }

    inline int parseNextOp(const char *ptr, char &op, int &num)
    {
      num = 0;
      const char * begin = ptr;
      for (char curChar = ptr[0]; curChar != 0; curChar = (++ptr)[0])
      {
        int digit = curChar - '0';
        if (digit >= 0 && digit <= 9) num = num*10 + digit;
        else break;
      }
      op = (ptr++)[0];
      return ptr - begin;
    }
   
    Status CalculatePosition(const AlignmentResult *result, const char* cigar, size_t cigar_len,
        int32_t &position) {
      // figure out the 5' position
      // the result->location is already genome relative, we shouldn't have to worry 
      // about negative index after clipping, but double check anyway
      // cigar parsing adapted from samblaster
      int ralen = 0, qalen = 0, sclip = 0, eclip = 0;
      bool first = true;
      char op;
      int op_len;
      while(cigar_len > 0) {
        size_t len = parseNextOp(cigar, op, op_len);
        cigar += len;
        cigar_len -= len;         
        LOG(INFO) << "cigar was " << op_len << " " << op;
        if (op == 'M' || op == '=' || op == 'X')
        {
          ralen += op_len;
          qalen += op_len;
          first = false;
        }
        else if (op == 'S' || op == 'H')
        {
          if (first) sclip += op_len;
          else       eclip += op_len;
        }
        else if (op == 'D' || op == 'N')
        {
          ralen += op_len;
        }
        else if (op == 'I')
        {
          qalen += op_len;
        }
        else
        {
          return Internal("Unknown opcode ", string(&op, 1), " in CIGAR string: ", string(cigar, cigar_len));
        }
      }
      if (IsForwardStrand(result)) {
        position = (int32_t)(result->location_ - sclip);
      } else {
        // im not 100% sure this is correct ...
        // but if it goes for every signature then it shouldn't matter
        position = (int32_t)(result->location_ + ralen + eclip - 1);
      }
      if (position < 0)
        return Internal("A position after applying clipping was < 0!");
      return Status::OK();
    }

    Status MarkDuplicate(const AlignmentResult* result, const char* cigar,
        size_t cigar_len, AlignmentResultBuilder &builder) {
      AlignmentResult result_out = *result; // simple copy suffices
      result_out.flag_ = result_out.flag_ | ResultFlag::PCR_DUPLICATE;
      // yes we are copying and rebuilding the entire structure
      // modifying in place is a huge pain in the ass, and the results aren't that
      // big anyway
      builder.AppendAlignmentResult(result_out, string(cigar, cigar_len));
      return Status::OK();
    }

    Status ProcessOrphan(const AlignmentResult* result, const char* cigar,
        size_t cigar_len, AlignmentResultBuilder &builder) {
      Signature sig;
      sig.is_forward = IsForwardStrand(result);
      TF_RETURN_IF_ERROR(CalculatePosition(result, cigar, cigar_len, sig.position));

      // attempt to find the signature
      auto sig_map_iter = signature_map_->find(sig);
      if (sig_map_iter == signature_map_->end()) { // not found, insert it
        signature_map_->insert(make_pair(sig, 1));
        // its the first here, others will be marked dup
        builder.AppendAlignmentResult(*result, string(cigar, cigar_len));
        return Status::OK();
      } else { 
        // found, mark a dup
        return MarkDuplicate(result, cigar, cigar_len, builder);
      }
    }

    void Compute(OpKernelContext* ctx) override {
      if (!bufferlist_pool_) {
        OP_REQUIRES_OK(ctx, InitHandles(ctx));
      }

      const Tensor* results_t, *num_results_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_results_t));
      OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_t));
      auto results_handle = results_t->vec<string>();
      auto num_results = num_results_t->scalar<int32>()();
      auto rmgr = ctx->resource_manager();
      ResourceContainer<Data> *results_container;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results_handle(0), results_handle(1), &results_container));
      AGDResultReader results_reader(results_container, num_results);

      // get output buffer pairs (pair holds [index, data] to construct
      // the results builder for output
      ResourceContainer<BufferList> *output_bufferlist_container;
      OP_REQUIRES_OK(ctx, GetOutputBufferList(ctx, &output_bufferlist_container));
      auto output_bufferlist = output_bufferlist_container->get();
      output_bufferlist->resize(1);
      AlignmentResultBuilder results_builder;
      results_builder.set_buffer_pair(&(*output_bufferlist)[3]);


      const AlignmentResult* result;
      const AlignmentResult* mate;
      const char* result_cigar, *mate_cigar;
      size_t result_cigar_len, mate_cigar_len;
      Status s = results_reader.GetNextResult(&result, &result_cigar, &result_cigar_len);

      // this detection logic adapted from SamBlaster
      while (s.ok()) {
        if (!IsPrimary(result))
          OP_REQUIRES_OK(ctx, Internal("Non-primary result detected in primary result column at location ",
                result->location_));
        if (!IsPaired(result)) {
          // we have a single alignment
          if (IsUnmapped(result)) {
            s = results_reader.GetNextResult(&result, &result_cigar, &result_cigar_len);
            continue;
          }

          OP_REQUIRES_OK(ctx, ProcessOrphan(result, result_cigar, result_cigar_len, results_builder));

        } else { // we have a pair, get the mate
          OP_REQUIRES_OK(ctx, results_reader.GetNextResult(&mate, &mate_cigar, &mate_cigar_len));

          OP_REQUIRES(ctx, (result->next_location_ == mate->location_) && (mate->next_location_ == result->location_),
              Internal("Malformed pair or the data is not in metadata (QNAME) order"));

          if (IsUnmapped(result) && IsUnmapped(mate)) {
            s = results_reader.GetNextResult(&result, &result_cigar, &result_cigar_len);
            continue;
          }
          
          if (IsUnmapped(result) && IsMapped(mate)) { // treat as single 
            OP_REQUIRES_OK(ctx, ProcessOrphan(mate, mate_cigar, mate_cigar_len, results_builder));
          } else if (IsUnmapped(mate) && IsMapped(result)) {
            OP_REQUIRES_OK(ctx, ProcessOrphan(result, result_cigar, result_cigar_len, results_builder));
          } else {
            Signature sig;
            sig.is_forward = IsForwardStrand(result);
            sig.is_mate_forward = IsForwardStrand(mate);
            OP_REQUIRES_OK(ctx, CalculatePosition(result, result_cigar, result_cigar_len, sig.position));
            OP_REQUIRES_OK(ctx, CalculatePosition(mate, mate_cigar, mate_cigar_len, sig.position_mate));

            // attempt to find the signature
            auto sig_map_iter = signature_map_->find(sig);
            if (sig_map_iter == signature_map_->end()) { // not found, insert it
              signature_map_->insert(make_pair(sig, 1));
              results_builder.AppendAlignmentResult(*result, string(result_cigar, result_cigar_len));
              results_builder.AppendAlignmentResult(*mate, string(mate_cigar, mate_cigar_len));
            } else { 
              // found, mark a dup
              LOG(INFO) << "omg we found a duplicate";
              OP_REQUIRES_OK(ctx, MarkDuplicate(result, result_cigar, result_cigar_len, results_builder));
              OP_REQUIRES_OK(ctx, MarkDuplicate(mate, mate_cigar, mate_cigar_len, results_builder));
            }
          }
        }
        s = results_reader.GetNextResult(&result, &result_cigar, &result_cigar_len);
      } // while s is ok()

      // done
      resource_releaser(results_container);
      LOG(INFO) << "DONE running mark duplicates!!";

    }

  private:
    ReferencePool<BufferList> *bufferlist_pool_ = nullptr;

    struct Signature {
      int32 position = 0;
      int32 position_mate = 0;
      bool is_forward = true;
      bool is_mate_forward = true;
      bool operator==(const Signature& s) {
        return (s.position == position) && (s.position_mate == position_mate)
          && (s.is_forward == is_forward) && (s.is_mate_forward == is_mate_forward);
      }
    };

    struct EqSignature {
      bool operator()(Signature sig1, Signature sig2) const {
        return sig1 == sig2;
      }
    };

    struct SigHash {
      size_t operator()(Signature const& s) const {
        size_t p = hash<int32>{}(s.position);
        size_t pm = hash<int32>{}(s.position_mate);
        size_t i = hash<bool>{}(s.is_forward);
        size_t m = hash<bool>{}(s.is_mate_forward);
        // maybe this is too expensive
        boost::hash_combine(p, pm);
        boost::hash_combine(i, m);
        boost::hash_combine(p, i);
        return p;
      }
    };

    typedef google::dense_hash_map<Signature, int, SigHash, EqSignature> SignatureMap;
    SignatureMap* signature_map_;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDMarkDuplicate").Device(DEVICE_CPU), AGDMarkDuplicateOp);
} //  namespace tensorflow {
