//
// Created by Saket Dingliwal on 12/07/17.
//

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/column_builder.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_record_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/compression.h"
#include "tensorflow/contrib/persona/kernels/agd-format/util.h"
#include "tensorflow/contrib/persona/kernels/agd-format/buffer.h"
#include <vector>
#include <cstdint>
#include <queue>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"
#include "tensorflow/core/framework/queue_interface.h"
#include <boost/functional/hash.hpp>
#include <google/dense_hash_map>
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "BamAlignment.h"
#include "BamRecord.h"



namespace tensorflow {

  namespace {
    typedef pair<AGDRecordReader* , AGDRecordReader*> BaseQualPair;
    typedef pair<AGDResultReader*,BaseQualPair> ResBaseQual;
    typedef pair<ResBaseQual, int> ReaderPair ;
    typedef pair<Alignment*, ReaderPair > ResultPair;

  }
  inline bool operator>(const Position& lhs, const Position& rhs) {
    if (lhs.ref_index() > rhs.ref_index()) {
      return true;
    } else if (lhs.ref_index() == rhs.ref_index()) {
      if (lhs.position() > rhs.position()) return true;
      else return false;
    } else
      return false;
  }
  struct PositionComparator {
    bool operator()(const ResultPair &a, const ResultPair &b) {
    return a.first->position() > b.first->position();
    }
  };



  class AGDMultiReader {


  public:
    AGDMultiReader();
    AGDMultiReader(OpKernelContext* ctx,int vec_size);
    ~AGDMultiReader();
    bool getNextAlignment(BamTools::BamAlignment& nextAlignment ) ;
    bool getNextAlignment(SeqLib::BamRecord& nextRecord);

  private:

    int vec_size;
    OpKernelContext* ctx;
    Status Init(OpKernelContext *ctx,int i);
    Status LoadDataResource(OpKernelContext* ctx, const Tensor* handle_t,ResourceContainer<Data>** container);
    void PQInit(OpKernelContext* ctx);
    Status DequeueElement(OpKernelContext *ctx,int i);
    vector<QueueInterface *> input_queue_ ;
    const Tensor *result_t, *base_t, *quality_t,*meta_t,*num_records_t;
    priority_queue<ResultPair, vector<ResultPair>, PositionComparator > multiReader ;
    vector<AGDResultReader> results_reader_global;
    vector<AGDRecordReader> base_reader_global;
    vector<AGDRecordReader> qual_reader_global;
    vector<Alignment> result_global;
    int chunk_count=0;


  };
}

