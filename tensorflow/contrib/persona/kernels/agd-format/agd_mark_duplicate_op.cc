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
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include <boost/functional/hash.hpp>
#include <google/dense_hash_map>

namespace tensorflow {

   namespace { 
      void resource_releaser(ResourceContainer<Data> *data) {
        core::ScopedUnref a(data);
        data->release();
      }
   }


  using namespace std;
  using namespace errors;
  using namespace format;

  class AGDMarkDuplicatesOp : public OpKernel {
  public:
    AGDMarkDuplicatesOp(OpKernelConstruction *context) : OpKernel(context) {
      signature_map_ = new SignatureMap();
      Signature sig;
      signature_map_->set_empty_key(sig);
    }

    ~AGDMarkDuplicatesOp() {
      LOG(INFO) << "Found a total of " << num_dups_found_ << " duplicates.";
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
        uint32_t &position) {
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
        //LOG(INFO) << "cigar was " << op_len << " " << op;
        //LOG(INFO) << "cigar len is now: " << cigar_len;
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
      //LOG(INFO) << "the location is: " << result->location_;
      if (IsForwardStrand(result)) {
        position = static_cast<uint32_t>(result->location_ - sclip);
      } else {
        // im not 100% sure this is correct ...
        // but if it goes for every signature then it shouldn't matter
        position = static_cast<uint32_t>(result->location_ + ralen + eclip - 1);
      }
      //LOG(INFO) << "position is now: " << position;
      if (position < 0)
        return Internal("A position after applying clipping was < 0! --> ", position);
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

      //LOG(INFO) << "sig is: " << sig.ToString();
      // attempt to find the signature
      auto sig_map_iter = signature_map_->find(sig);
      if (sig_map_iter == signature_map_->end()) { // not found, insert it
        signature_map_->insert(make_pair(sig, 1));
        // its the first here, others will be marked dup
        builder.AppendAlignmentResult(*result, string(cigar, cigar_len));
        return Status::OK();
      } else { 
        // found, mark a dup
        num_dups_found_++;
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
      results_builder.set_buffer_pair(&(*output_bufferlist)[0]);


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

          //LOG(INFO) << "processing mapped orphan at " << result->location_;
          OP_REQUIRES_OK(ctx, ProcessOrphan(result, result_cigar, result_cigar_len, results_builder));

        } else { // we have a pair, get the mate
          OP_REQUIRES_OK(ctx, results_reader.GetNextResult(&mate, &mate_cigar, &mate_cigar_len));

          OP_REQUIRES(ctx, (result->next_location_ == mate->location_) && (mate->next_location_ == result->location_),
              Internal("Malformed pair or the data is not in metadata (QNAME) order. At index: ", results_reader.GetCurrentIndex()-1,
                "result 1: ", result->ToString(), " result2: ", mate->ToString()));

          //LOG(INFO) << "processing mapped pair at " << result->location_ << ", " << mate->location_;

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
              num_dups_found_++;
            }
          }
        }
        s = results_reader.GetNextResult(&result, &result_cigar, &result_cigar_len);
      } // while s is ok()

      // done
      resource_releaser(results_container);
      LOG(INFO) << "DONE running mark duplicates!! Found so far: " << num_dups_found_;

    }

  private:
    ReferencePool<BufferList> *bufferlist_pool_ = nullptr;

    struct Signature {
      uint32_t position = 0;
      uint32_t position_mate = 0;
      bool is_forward = true;
      bool is_mate_forward = true;
      bool operator==(const Signature& s) {
        return (s.position == position) && (s.position_mate == position_mate)
          && (s.is_forward == is_forward) && (s.is_mate_forward == is_mate_forward);
      }
      string ToString() const {
        return string("pos: ") + to_string(position) + " matepos: " + to_string(position_mate)
          + " isfor: " + to_string(is_forward) + " ismatefor: " + to_string(is_mate_forward) ;
      }
    };

    struct EqSignature {
      bool operator()(Signature sig1, Signature sig2) const {
        return (sig1.position == sig2.position) && (sig1.position_mate == sig2.position_mate) 
          && (sig1.is_forward == sig2.is_forward) && (sig1.is_mate_forward == sig2.is_mate_forward);
      }
    };

    struct SigHash {
      size_t operator()(Signature const& s) const {
        size_t p = hash<uint32_t>{}(s.position);
        size_t pm = hash<uint32_t>{}(s.position_mate);
        size_t i = hash<bool>{}(s.is_forward);
        size_t m = hash<bool>{}(s.is_mate_forward);
        // maybe this is too expensive
        boost::hash_combine(p, pm);
        boost::hash_combine(i, m);
        boost::hash_combine(p, i);
        //LOG(INFO) << "hash was called on " << s.ToString() << " and value was: " << p;
        return p;
      }
    };

    typedef google::dense_hash_map<Signature, int, SigHash, EqSignature> SignatureMap;
    SignatureMap* signature_map_;

    int num_dups_found_ = 0;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDMarkDuplicates").Device(DEVICE_CPU), AGDMarkDuplicatesOp);
} //  namespace tensorflow {
