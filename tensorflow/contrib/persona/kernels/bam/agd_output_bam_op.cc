
// Stuart Byma
// Using BAM format headers/processing from SNAP

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/persona/kernels/object-pool/basic_container.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include <sstream>
#include <tuple>
#include <thread>
#include <vector>
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"
#include "tensorflow/contrib/persona/kernels/object-pool/ref_pool.h"
#include "tensorflow/contrib/persona/kernels/agd-format/agd_result_reader.h"
#include "tensorflow/contrib/persona/kernels/agd-format/format.h"
#include "tensorflow/contrib/persona/kernels/agd-format/sam_flags.h"
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
#include "tensorflow/contrib/persona/kernels/snap-align/snap/SNAPLib/Bam.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/concurrent_queue.h"
#include "tensorflow/contrib/persona/kernels/concurrent_queue/priority_concurrent_queue.h"
#include "zlib.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  
  using shape_inference::InferenceContext;
   
  namespace { 
    void resource_releaser(ResourceContainer<Data> *data) {
      core::ScopedUnref a(data);
      data->release();
    }
  }


  class AgdOutputBamOp : public OpKernel {
    public:
      explicit AgdOutputBamOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    
        string path, pg_id, read_group, sort_order;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("path", &path));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("pg_id", &pg_id));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("read_group", &read_group));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ref_sequences", &ref_seqs_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ref_seq_sizes", &ref_sizes_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("sort_order", &sort_order));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
        LOG(INFO) << "using " << num_threads_ << " threads";

        OP_REQUIRES(ctx, ref_seqs_.size() == ref_sizes_.size(), 
            Internal("ref seqs was not same size as ref seq sizes lists"));
        stringstream header_ss;
        header_ss << "@HD\tVN:1.4\tSO:";
        header_ss << sort_order << endl;
        ref_size_totals_.reserve(ref_seqs_.size());
        int64_t total = 0;
        for (int i = 0; i < ref_seqs_.size(); i++) {
          total += ref_sizes_[i];
          ref_size_totals_.push_back(total);
        }
        // open the file, we dont write yet
        bam_fp_ = fopen(path.c_str(), "w");
        OP_REQUIRES(ctx, bam_fp_ != NULL );
        header_ = header_ss.str();

        buffer_queue_.reset(new ConcurrentQueue<BufferRef>(num_threads_*2));
        compress_queue_.reset(new ConcurrentQueue<CompressItem>(num_threads_*2));
        write_queue_.reset(new PriorityConcurrentQueue<WriteItem>(num_threads_*2));

        // *2 because we need an output buf for compress and write
        // for each input buf
        buffers_.resize(num_threads_*2);
        for (int i = 0; i < num_threads_*2; i++) {
          buffers_[i].reset(new char[buffer_size_]);
          buffer_queue_->push(&buffers_[i]);
        }
        
        compute_status_ = Status::OK();
      }

      ~AgdOutputBamOp() override {
        // drain the queues first
        while (compress_queue_->size() > 0) {
          this_thread::sleep_for(chrono::milliseconds(10));
        }
       
        // now stop the threads
        run_compress_ = false;
        compress_queue_->unblock();
        //LOG(INFO) << "Stopping c threads ...";
        while (num_active_threads_.load() > 1) {
          this_thread::sleep_for(chrono::milliseconds(10));
        }

        // it may be that the compress threads give one last 
        // block to write, we dont want to lose it, so stop the writer
        // after
        while (write_queue_->size() > 0) {
          this_thread::sleep_for(chrono::milliseconds(10));
        }
        run_write_ = false;
        write_queue_->unblock();
        while (num_active_threads_.load() > 0) {
          this_thread::sleep_for(chrono::milliseconds(10));
        }

        // if we have a partial buffer, compress and write it out

        if (scratch_pos_ != 0) {
          BufferRef buf;
          buffer_queue_->pop(buf);
          size_t compressed_size = 0;
          Status s = CompressToBuffer(scratch_, scratch_pos_, buf->get(), buffer_size_, compressed_size);
          if (!s.ok() || compressed_size == 0) {
            LOG(ERROR) << "Error in final compress and write, compressed size = " << compressed_size;
          }
          int status = fwrite(buf->get(), compressed_size, 1, bam_fp_);

          if (status < 0) 
            LOG(INFO) << "WARNING: Final write of BAM failed with " << status;
        }
        // EOF marker for bam BGZF format
        static _uint8 eof[] = {
          0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43,
          0x02, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
        };
       
        // append the final EOF marker
        int status = fwrite(eof, sizeof(eof), 1, bam_fp_);
        
        if (status < 0) 
          LOG(INFO) << "WARNING: Final write of BAM eof marker failed with " << status;

        status = fclose(bam_fp_);
        
        if (status == EOF) 
          LOG(INFO) << "WARNING: Failed to close BAM file pointer: " << status;

        /*LOG(INFO) << "compress queue push wait: " << compress_queue_->num_push_waits();
        LOG(INFO) << "write queue push wait: " << write_queue_->num_push_waits();
        LOG(INFO) << "compress queue pop wait: " << compress_queue_->num_pop_waits();
        LOG(INFO) << "write queue pop wait: " << write_queue_->num_pop_waits();
        LOG(INFO) << "buffer queue push wait: " << buffer_queue_->num_push_waits();
        LOG(INFO) << "buffer queue pop wait: " << buffer_queue_->num_pop_waits();*/
      }

      void Compute(OpKernelContext* ctx) override {
        if (first_) {
          Init(ctx);
          //LOG(INFO) << "getting buffer";
          buffer_queue_->pop(current_buf_ref_);
          scratch_ = current_buf_ref_->get();
          scratch_pos_ = 0;
          // and write the header into the first buffer
          BAMHeader* bamHeader = (BAMHeader*) scratch_;
          bamHeader->magic = BAMHeader::BAM_MAGIC;
          size_t samHeaderSize = header_.length();
          memcpy(bamHeader->text(), header_.c_str(), samHeaderSize);
          bamHeader->l_text = (int)samHeaderSize;
          scratch_pos_ = BAMHeader::size((int)samHeaderSize);

          bamHeader->n_ref() = ref_seqs_.size();
          BAMHeaderRefSeq* refseq = bamHeader->firstRefSeq();
          for (int i = 0; i < ref_seqs_.size(); i++) {
            int len = ref_seqs_[i].length() + 1;
            scratch_pos_ += BAMHeaderRefSeq::size(len);
            refseq->l_name = len;
            memcpy(refseq->name(), ref_seqs_[i].c_str(), len);
            refseq->l_ref() = (int)(ref_sizes_[i]);
            refseq = refseq->next();
            _ASSERT((char*) refseq - header == scratch_pos_);
          }
          first_ = false;
        }

        const Tensor *results_in, *bases_in, *qualities_in, *metadata_in, *num_records_t;
        OP_REQUIRES_OK(ctx, ctx->input("num_records", &num_records_t));
        OP_REQUIRES_OK(ctx, ctx->input("results_handle", &results_in));
        OP_REQUIRES_OK(ctx, ctx->input("bases_handle", &bases_in));
        OP_REQUIRES_OK(ctx, ctx->input("qualities_handle", &qualities_in));
        OP_REQUIRES_OK(ctx, ctx->input("metadata_handle", &metadata_in));
       
        auto num_records = num_records_t->scalar<int32>()();

        ResourceContainer<Data>* bases_data, *qual_data, *meta_data;
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, bases_in, &bases_data));
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, qualities_in, &qual_data));
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, metadata_in, &meta_data));

        AGDRecordReader base_reader(bases_data, num_records);
        AGDRecordReader qual_reader(qual_data, num_records);
        AGDRecordReader meta_reader(meta_data, num_records);
        vector<unique_ptr<AGDResultReader>> result_readers;
        vector<ResourceContainer<Data>*> results_data;

        results_data.resize(results_in->dim_size(0));
        for (size_t i = 0; i < results_in->dim_size(0); i++) {

          OP_REQUIRES_OK(ctx, LoadDataResource(ctx, i, results_in, &results_data[i]));
          result_readers.push_back(
                  unique_ptr<AGDResultReader>(new AGDResultReader(results_data[i], num_records)));
        }

        Alignment result;
        const char* data, *meta, *base, *qual;
        const char* cigar;
        size_t meta_len, base_len, qual_len, cigar_len;
        int ref_index, mate_ref_index;
        vector<uint32> cigar_vec;
        cigar_vec.reserve(20); // should usually be enough

        Status s = Status::OK();
        while (s.ok()) {
          OP_REQUIRES_OK(ctx, meta_reader.GetNextRecord(&meta, &meta_len));
          OP_REQUIRES_OK(ctx, base_reader.GetNextRecord(&base, &base_len));
          OP_REQUIRES_OK(ctx, qual_reader.GetNextRecord(&qual, &qual_len));
          // cut off the metadata, it can't have spaces apparently
          const char* occ = strchr(meta, ' ');
          if (occ) 
            meta_len = occ - meta;

          //LOG(INFO) << "processing record " << string(meta, meta_len);

          // write an entry for each result, skip empty secondaries
          for (uint32 i = 0; i < result_readers.size(); i++) {
            //OP_REQUIRES_OK(ctx, result_readers[i]->GetNextResult(result));
            s = result_readers[i]->GetNextResult(result);
            OP_REQUIRES(ctx, i == 0 && s.ok() || i > 0 && (s.ok() || IsUnavailable(s)),
                        Internal("Output bam received bad alignment result"))
            if (IsUnavailable(s)) {
              // skip emtpy secondary
              continue;
            }

            cigar = result.cigar().c_str();
            cigar_len = result.cigar().length();
            OP_REQUIRES_OK(ctx, ParseCigar(cigar, cigar_len, cigar_vec));

            size_t bamSize = BAMAlignment::size((unsigned) meta_len + 1, cigar_vec.size(), base_len, /*auxLen*/0);
            if ((buffer_size_ - scratch_pos_) < bamSize) {
              // full buffer, push to compress queue and get a new buffer
              //LOG(INFO) << "main is getting buf for compress";
              BufferRef compress_ref;
              buffer_queue_->pop(compress_ref);
              //LOG(INFO) << "main is pushing to compress";
              compress_queue_->push(
                      make_tuple(current_buf_ref_, scratch_pos_, compress_ref, buffer_size_, current_index_));

              //LOG(INFO) << "main is getting fresh buf";
              current_index_++;
              buffer_queue_->pop(current_buf_ref_);
              scratch_ = current_buf_ref_->get();
              scratch_pos_ = 0;
            }

            BAMAlignment *bam = (BAMAlignment *) (scratch_ + scratch_pos_);
            bam->block_size = (int) bamSize - 4;

            bam->refID = result.position().ref_index();
            bam->pos = result.position().position();
            bam->l_read_name = (_uint8) meta_len + 1;
            bam->MAPQ = result.mapping_quality();
            bam->next_refID = result.next_position().ref_index();
            bam->next_pos = result.next_position().position();

            int refLength = cigar_vec.size() > 0 ? 0 : base_len;
            for (int i = 0; i < cigar_vec.size(); i++) {
              refLength += BAMAlignment::CigarCodeToRefBase[cigar_vec[i] & 0xf] * (cigar_vec[i] >> 4);
            }

            if (IsUnmapped(result.flag())) {
              if (IsNextUnmapped(result.flag())) {
                bam->bin = BAMAlignment::reg2bin(-1, 0);
              } else {
                bam->bin = BAMAlignment::reg2bin(bam->next_pos, bam->next_pos + 1);
              }
            } else {
              bam->bin = BAMAlignment::reg2bin(bam->pos, bam->pos + refLength);
            }

            bam->n_cigar_op = cigar_vec.size();
            bam->FLAG = result.flag();
            bam->l_seq = base_len;
            bam->tlen = (int) result.template_length();
            memcpy(bam->read_name(), meta, meta_len);
            bam->read_name()[meta_len] = 0;
            memcpy(bam->cigar(), &cigar_vec[0], cigar_vec.size() * 4);
            BAMAlignment::encodeSeq(bam->seq(), base, base_len);
            memcpy(bam->qual(), qual, qual_len);
            for (unsigned i = 0; i < qual_len; i++) {
              bam->qual()[i] -= '!';
            }
            bam->validate();

            scratch_pos_ += bamSize;

           // s = result_readers.GetNextRecord(&data, &len);
          }

          s = result_readers[0]->PeekNextResult(result);
        }

        resource_releaser(bases_data);
        resource_releaser(qual_data);
        resource_releaser(meta_data);
        for (auto d : results_data)
          resource_releaser(d);


        Tensor* out_t;
        OP_REQUIRES_OK(ctx, ctx->allocate_output("chunk", TensorShape({}), &out_t));
        auto& out = out_t->scalar<int32>()();
        out = count_;
        count_++;

      }

    private:

      Status CompressToBuffer(char* in_buf, uint32_t in_size, char* out_buf, uint32_t out_size,
          size_t &compressed_size) {
        if (in_size == 0)
          return Internal("attempting to compress and write 0 bytes");
        // set up BAM header structure
        gz_header header;
        _uint8 bamExtraData[6];
        header.text = false;
        header.time = 0;
        header.xflags = 0;
        header.os = 0;
        header.extra = bamExtraData;
        header.extra_len = 6;
        header.extra_max = 6;
        header.name = NULL;
        header.name_max = 0;
        header.comment = NULL;
        header.comm_max = 0;
        header.hcrc = false;
        header.done = true;
        bamExtraData[0] = 'B';
        bamExtraData[1] = 'C';
        bamExtraData[2] = 2;
        bamExtraData[3] = 0;
        bamExtraData[4] = 3; // will be filled in later
        bamExtraData[5] = 7; // will be filled in later

        z_stream zstream;
        zstream.zalloc = Z_NULL;
        zstream.zfree = Z_NULL;
        const int windowBits = 15;
        const int GZIP_ENCODING = 16;
        zstream.next_in = (Bytef*) in_buf;
        zstream.avail_in = (uInt)in_size;
        zstream.next_out = (Bytef*) out_buf;
        zstream.avail_out = (uInt)out_size;
        uInt oldAvail;
        int status;

        status = deflateInit2(&zstream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, windowBits | GZIP_ENCODING, 8, Z_DEFAULT_STRATEGY);
        if (status < 0) 
          return Internal("libz deflate init failed with ", to_string(status));

        status = deflateSetHeader(&zstream, &header);
        if (status != Z_OK) {
          return Internal("libz: defaultSetHeader failed with", status);
        }

        oldAvail = zstream.avail_out;
        status = deflate(&zstream, Z_FINISH);

        if (status < 0 && status != Z_BUF_ERROR) {
          return Internal("libz: deflate failed with ", status);
        }
        if (zstream.avail_in != 0) {
          return Internal("libz: default failed to read all input");
        }
        if (zstream.avail_out == oldAvail) {
          return Internal("libz: default failed to write output");
        }
        status = deflateEnd(&zstream);
        if (status < 0) {
          return Internal("libz: deflateEnd failed with ", status);
        }

        size_t toUsed = out_size - zstream.avail_out;
        // backpatch compressed block size into gzip header
        if (toUsed >= BAM_BLOCK) {
          return Internal("exceeded BAM chunk size");
        }
        * (_uint16*) (out_buf + 16) = (_uint16) (toUsed - 1);

        //status = fwrite(scratch_compress_.get(), toUsed, 1, bam_fp_);

        //if (status < 0)
          //return Internal("write compressed block to bam failed: ", status);

        if (!((BgzfHeader*)(out_buf))->validate(toUsed, in_size)) {
          return Internal("bgzf validation failed");
        }

        compressed_size = toUsed;
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

      Status ParseCigar(const char* cigar, size_t cigar_len, vector<uint32>& cigar_vec) {
        // cigar parsing adapted from samblaster
        cigar_vec.clear();
        char op;
        int op_len;
        while(cigar_len > 0) {
          size_t len = parseNextOp(cigar, op, op_len);
          cigar += len;
          cigar_len -= len;
          uint32 val = (op_len << 4) | BAMAlignment::CigarToCode[op];
          cigar_vec.push_back(val);
        }
        return Status::OK();
      }

      Status LoadDataResource(OpKernelContext* ctx, const Tensor* handle_t,
          ResourceContainer<Data>** container) {
        auto rmgr = ctx->resource_manager();
        auto handles_vec = handle_t->vec<string>();

        TF_RETURN_IF_ERROR(rmgr->Lookup(handles_vec(0), handles_vec(1), container));
        return Status::OK();
      }

      Status LoadDataResource(OpKernelContext* ctx, uint32 index, const Tensor* handle_t,
                              ResourceContainer<Data>** container) {
        auto rmgr = ctx->resource_manager();
        auto handles_mat = handle_t->matrix<string>();

        TF_RETURN_IF_ERROR(rmgr->Lookup(handles_mat(index, 0), handles_mat(index, 1), container));
        return Status::OK();
      }

      void Init(OpKernelContext* ctx) {

        auto compress_func = [this] () {
          
          /*int my_id = 0;
          {
            mutex_lock l(mu_);
            my_id = thread_id_++;
          }*/
        
          size_t compressed_size;
          CompressItem item;
          while (run_compress_) {
            
            if (!compress_queue_->pop(item)) {
              continue;
            }

            BufferRef in_buf = get<0>(item);
            auto in_size = get<1>(item); 
            BufferRef out_buf = get<2>(item);
            auto out_size = get<3>(item); 
            auto index = get<4>(item); 

            compressed_size = 0;
            //LOG(INFO) << my_id <<  " compressor compressing index " << index << " at size " << in_size << " bytes to "
             // << " output buf with size " << out_size;
            Status s = CompressToBuffer(in_buf->get(), in_size, out_buf->get(), out_size, compressed_size);
            if (!s.ok()) {
              LOG(ERROR) << "Error in compress and write";
              compute_status_ = s;
              return;
            }
            //LOG(INFO) << my_id << " compressed into " << compressed_size << " bytes.";

            buffer_queue_->push(in_buf); // recycle buffer
            WriteItem wr_item;
            wr_item.buf = out_buf;
            wr_item.size = compressed_size;
            wr_item.index = index;
            //LOG(INFO)<< my_id  << " compressor pushing " << index << " to writer ";
            write_queue_->push(wr_item);
          }
          num_active_threads_--;
        };

        auto writer_func = [this] () {
          /*int my_id = 0;
          {
            mutex_lock l(mu_);
            my_id = thread_id_++;
          }*/
     
          uint32_t index = 0;
          WriteItem item;
          while (run_write_) {
            
            if (!write_queue_->peek(item)) {
              continue;
            }
            // its possible the next index is not in the queue yet
            // wait for it
            // only works if there is one thread doing this
            if (item.index != index)
              continue;

            // got the right index now, works because
            // we know a lower index than `index` will never
            // happen
            //LOG(INFO) << my_id << " writer popping";
            write_queue_->pop(item);

            auto* buf = item.buf;
            auto size = item.size;
            auto idx = item.index;

            //LOG(INFO) << my_id << " writer writing index " << idx;
            if (idx != index) {
              //LOG(INFO) << my_id << " got " << idx << " for index " << index;
              compute_status_ = Internal("Did not get block index in order!");
              return;
            }

            int status = fwrite(buf->get(), size, 1, bam_fp_);
            if (status < 0) {
              compute_status_ = Internal("Failed to write to bam file");
              return;
            }
            index++;

            // recycle the buffer
            //LOG(INFO) << my_id << "writer recycling";
            buffer_queue_->push(buf);
          }
          num_active_threads_--;
        };

        auto worker_threadpool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
        for (int i = 0; i < num_threads_-1; i++)
          worker_threadpool->Schedule(compress_func);
        worker_threadpool->Schedule(writer_func);

        num_active_threads_ = num_threads_;
      }

      vector<string> ref_seqs_;
      vector<int32> ref_sizes_;
      vector<int64> ref_size_totals_;
      int32 count_ = 0;
      string header_;
  
      mutex mu_;
      bool first_ = true;
      uint32_t current_index_ = 0;
      //unique_ptr<char> scratch_, scratch_compress_;
      //uint64_t scratch_pos_ = 0;
      const uint64_t buffer_size_ = 64*1024; // 64Kb
      FILE* bam_fp_ = nullptr;
      
      volatile bool run_compress_ = true;
      volatile bool run_write_ = true;
      atomic<uint32_t> num_active_threads_;
      int thread_id_ = 0;
      int num_threads_;

      typedef const unique_ptr<char[]>* BufferRef;
      unique_ptr<ConcurrentQueue<BufferRef>> buffer_queue_;

      typedef tuple<BufferRef, uint32_t, BufferRef, uint32_t, uint32_t> CompressItem; // inbuffer, isize, outbuffer, osize, index (ordering)
      unique_ptr<ConcurrentQueue<CompressItem>> compress_queue_;

      struct WriteItem {
        BufferRef buf;
        size_t size;
        uint32_t index;
        bool operator<(const WriteItem& other) const {
          return index > other.index;
        }
      };

      unique_ptr<PriorityConcurrentQueue<WriteItem>> write_queue_;


      vector<unique_ptr<char[]>> buffers_;

      BufferRef current_buf_ref_;
      char* scratch_;
      uint32_t scratch_pos_;

      Status compute_status_;
      TF_DISALLOW_COPY_AND_ASSIGN(AgdOutputBamOp);
  };


  REGISTER_KERNEL_BUILDER(Name("AgdOutputBam").Device(DEVICE_CPU), AgdOutputBamOp);

}  // namespace tensorflow
