#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <vector>
#include <cstdint>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"
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

  inline bool operator==(const Position& lhs, const Position& rhs) {
    return (lhs.ref_index() == rhs.ref_index() && lhs.position() == rhs.position());
  }

  class AGDQualBinOp : public OpKernel {
  public:
    AGDQualBinOp(OpKernelConstruction *context) : OpKernel(context) {
      signature_map_ = new SignatureMap();
      Signature sig;
      signature_map_->set_empty_key(sig);
    }

    ~AGDQualBinOp() {
      core::ScopedUnref unref_listpool(bufferpair_pool_);
      delete signature_map_;
    }

    Status GetOutputBufferPair(OpKernelContext* ctx, ResourceContainer<BufferPair> **ctr)
    {
      TF_RETURN_IF_ERROR(bufferpair_pool_->GetResource(ctr));
      (*ctr)->get()->reset();
      TF_RETURN_IF_ERROR((*ctr)->allocate_output("marked_results", ctx));
      return Status::OK();
    }
    
    Status InitHandles(OpKernelContext* ctx)
    {
      TF_RETURN_IF_ERROR(GetResourceFromContext(ctx, "buffer_pair_pool", &bufferpair_pool_));

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
   
    Status CalculatePosition(const Alignment *result,
        uint32_t &position) {
      // figure out the 5' position
      // the result->location is already genome relative, we shouldn't have to worry 
      // about negative index after clipping, but double check anyway
      // cigar parsing adapted from samblaster
      const char* cigar;
      size_t cigar_len;
      cigar = result->cigar().c_str();
      cigar_len = result->cigar().length();
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
      if (IsForwardStrand(result->flag())) {
        position = static_cast<uint32_t>(result->position().position() - sclip);
      } else {
        // im not 100% sure this is correct ...
        // but if it goes for every signature then it shouldn't matter
        position = static_cast<uint32_t>(result->position().position() + ralen + eclip - 1);
      }
      //LOG(INFO) << "position is now: " << position;
      if (position < 0)
        return Internal("A position after applying clipping was < 0! --> ", position);
      return Status::OK();
    }

    void Compute(OpKernelContext* ctx) override {
      	LOG(INFO) << "helloworld";
	/*
	if (!bufferpair_pool_) {
        OP_REQUIRES_OK(ctx, InitHandles(ctx));
      }
      
	//set up
      const Tensor* results_t, *num_results_t;
      OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_results_t));
      OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_t));
      auto results_handle = results_t->vec<string>();
      auto num_results = num_results_t->scalar<int32>()();
      auto rmgr = ctx->resource_manager();
      
      ResourceContainer<Data> *record_container;
      OP_REQUIRES_OK(ctx, rmgr->Lookup(results_handle(0), results_handle(1), &record_container));
      

	AGDRecordReader record_reader(record_container, num_results);

      // get output buffer pairs (pair holds [index, data] to construct
      // the results builder for output
  
    ResourceContainer<BufferPair> *output_bufferpair_container;
      OP_REQUIRES_OK(ctx, GetOutputBufferPair(ctx, &output_bufferpair_container));
  
    auto output_bufferpair = output_bufferpair_container->get();
      AlignmentResultBuilder results_builder;
      results_builder.SetBufferPair(output_bufferpair);


      Alignment record;
      Status s = record_reader.GetNextRecord(record);

      // this detection logic adapted from SamBlaster
      while (s.ok()) {
	
        s = record_reader.GetNextRecord(record);
      } // while s is ok()

      // done
     /resource_releaser(record_container);
	*/
	
	OP_REQUIRES_OK(ctx, ctx->allocate_output("marked_results", ctx));
    }

  private:
    ReferencePool<BufferPair> *bufferpair_pool_ = nullptr;

    struct Signature {
      uint32_t position = 0;
      uint32_t ref_index = 0;
      uint32_t position_mate = 0;
      uint32_t ref_index_mate = 0;
      bool is_forward = true;
      bool is_mate_forward = true;
      bool operator==(const Signature& s) {
        return (s.position == position) && (s.position_mate == position_mate)
          && (s.is_forward == is_forward) && (s.is_mate_forward == is_mate_forward)
                && (s.ref_index == ref_index) && (s.ref_index_mate == ref_index_mate);
      }
      string ToString() const {
        return string("pos: ") + to_string(position) + " matepos: " + to_string(position_mate)
          + " isfor: " + to_string(is_forward) + " ismatefor: " + to_string(is_mate_forward) ;
      }
    };

    struct EqSignature {
      bool operator()(Signature sig1, Signature sig2) const {
        return (sig1.position == sig2.position) && (sig1.position_mate == sig2.position_mate) 
          && (sig1.is_forward == sig2.is_forward) && (sig1.is_mate_forward == sig2.is_mate_forward)
          && (sig1.ref_index == sig2.ref_index) && (sig1.ref_index_mate == sig2.ref_index_mate);
      }
    };

    struct SigHash {
      size_t operator()(Signature const& s) const {
        size_t p = hash<uint32_t>{}(s.position);
        size_t pm = hash<uint32_t>{}(s.position_mate);
        size_t ri = hash<uint32_t>{}(s.ref_index);
        size_t rim = hash<uint32_t>{}(s.ref_index_mate);
        size_t i = hash<bool>{}(s.is_forward);
        size_t m = hash<bool>{}(s.is_mate_forward);
        // maybe this is too expensive
        boost::hash_combine(ri, rim);
        boost::hash_combine(p, pm);
        boost::hash_combine(i, m);
        boost::hash_combine(p, i);
        boost::hash_combine(p, ri);
        //LOG(INFO) << "hash was called on " << s.ToString() << " and value was: " << p;
        return p;
      }
    };

    typedef google::dense_hash_map<Signature, int, SigHash, EqSignature> SignatureMap;
    SignatureMap* signature_map_;

    int num_dups_found_ = 0;

  };

  REGISTER_KERNEL_BUILDER(Name("AGDQualBinOp").Device(DEVICE_CPU), AGDQualBinOp);
} //  namespace tensorflow {
