#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/user_ops/dense-format/buffer.h"
#include "tensorflow/core/user_ops/object-pool/resource_container.h"
#include "GenomeIndex.h"
#include "genome_index_resource.h"
#include "Read.h"
#include "tensorflow/core/user_ops/dna-align/snap/SNAPLib/SAM.h"
#include <cstdint>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

namespace tensorflow {
  using namespace std;
  using namespace errors;

  REGISTER_OP("DenseTester")
  .Attr("container: string = ''")
  .Attr("shared_name: string = ''")
  .Attr("sam_filename: string = ''")
  .Input("dense_buffer: string")
  .Input("genome_handle: Ref(string)")
  .Output("result_buf_handle: string")
  .Doc(R"doc(
  Compares the dense format output with the SAM format output
)doc");

  class DenseTesterOp : public OpKernel {
  public:
    DenseTesterOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("sam_filename", &sam_filename));
    }
		
    _int64 getFileSize(const char *fileName) {
      struct stat sb;
      int fd = open(fileName, O_RDONLY);
      int r = fstat(fd, &sb);
      _int64 fileSize = sb.st_size;
      close(fd);
      return fileSize;
    }

    void Compute(OpKernelContext* ctx) override {

//    Get the dense format results in dense_buf
//    data(0) = container, data(1) = name
    	ResourceContainer<Data> *dense_buf;
      const Tensor *dense_input;
      OP_REQUIRES_OK(ctx, ctx->input("dense_buffer", &dense_input));
      auto dense_data = dense_input->vec<string>();
      OP_REQUIRES_OK(ctx, ctx->resource_manager()->Lookup(dense_data(0), dense_data(1), &dense_buf));

      OP_REQUIRES_OK(ctx,
          GetResourceFromContext(ctx, "genome_handle", &genome_resource_));
			
			// creating an output just to feed the BufferSinkOp, which should be the same as the "dense_buffer input"
      const TensorShape handle_shape_({2});
      Tensor *handle;
      OP_REQUIRES_OK(ctx, ctx->allocate_output("result_buf_handle", handle_shape_, &handle));
      auto handle_vec = handle->vec<string>();
      handle_vec(0) = dense_data(0);
      handle_vec(1) = dense_data(1); 
			
      const Genome *genome = genome_resource_->get_genome();
      
      // populate a context necessary for reading the SAM file
      ReaderContext readerContext;
      readerContext.genome = genome;
      readerContext.ignoreSecondaryAlignments = true;
      readerContext.ignoreSupplementaryAlignments = true;
      readerContext.clipping = NoClipping;
      readerContext.defaultReadGroup = "";
      readerContext.header = NULL;
      readerContext.headerLength = 0;
      readerContext.headerBytes = 0;

      _int64 sam_filesize = getFileSize(sam_filename.c_str());
      SAMReader *reader;
      reader = SAMReader::create(DataSupplier::Default, sam_filename.c_str(), 2, readerContext, 0, sam_filesize);
      
      // will be populated in SAMReader::getNextRead
      AlignmentResult alignmentResult;
      GenomeLocation genomeLocation;
      Direction direction; // do we need it? else = NULL
      unsigned mapQ;
      unsigned flag;
      const char *cigar;

      Read *read;
      int i = 0;
      while (reader->getNextRead(read, &alignmentResult, &genomeLocation, &direction, &mapQ, &flag, &cigar)) {
        std::cout << "The read with number " << i << " has the mapQ " << mapQ << std::endl;
        i++;
      }

/*
      const Genome *genome = genome_resource_->get_genome();
      ReaderContext readerContext;
      readerContext.genome = genome;
      readerContext.ignoreSecondaryAlignments = true;
      readerContext.ignoreSupplementaryAlignments = true;
      
      // TODO: what about the other parameters?
      readerContext.clipping = NoClipping;
      readerContext.defaultReadGroup = "";
      readerContext.header = NULL;
      readerContext.headerLength = 0;
      readerContext.headerBytes = 0;

      
      

      //      while (getNextRead(&sam_read_, alignmentResult, genomeLocation, direction, mapQ, flag, true, &cigar)) {
//          // do something
//      }

//      SAMReader::getNextRead(
//        Read *read, // this is where all the data will be - isSET
//        AlignmentResult *alignmentResult, // isSET (i think)
//        GenomeLocation *genomeLocation, // isSET
//        Direction *direction, // isSET
//        unsigned *mapQ, 
//        unsigned *flag,
//        bool ignoreEndOfRange, // not used anywhere??
//        const char **cigar)
*/
  }

  private:     
    string sam_filename;
    GenomeIndexResource* genome_resource_;
    ResourceContainer<Buffer> *sam_buf_;
    Read sam_read_;
  };


REGISTER_KERNEL_BUILDER(Name("DenseTester").Device(DEVICE_CPU), DenseTesterOp);
} // namespace tensorflow
