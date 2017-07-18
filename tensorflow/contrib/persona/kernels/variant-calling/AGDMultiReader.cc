//
// Created by Saket Dingliwal on 12/07/17.
//

#include "tensorflow/contrib/persona/kernels/variant-calling/AGDMultiReader.h"


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



  AGDMultiReader::AGDMultiReader(){
  }



  AGDMultiReader::AGDMultiReader(OpKernelContext* ctx,int vec_size_arg){
    this->ctx = ctx;
    vec_size = vec_size_arg;
    for(int i=0;i<vec_size;i++)
    {
      input_queue_.push_back(NULL);
    }
    for(int i=0;i<vec_size;i++){
      if (!input_queue_[i]) {
        OP_REQUIRES_OK(ctx, Init(ctx,i));
        cout << "queue initialized " <<i <<endl;
      }
    }
    PQInit(ctx);
  }
  AGDMultiReader::~AGDMultiReader(){
    for(int i=0;i<vec_size;i++)
    {
      core::ScopedUnref a1(input_queue_[i]);
    }
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



  bool AGDMultiReader::getNextAlignment(BamTools::BamAlignment& nextAlignment )
  {

    if(!multiReader.empty()){
      auto &top = multiReader.top();
      const char* base;
      size_t base_length;
      const char* qual;
      size_t qual_length;
      Alignment* result = top.first ;
      Status sb = top.second.first.second.first->GetNextRecord(&base,&base_length);
      Status sq = top.second.first.second.second->GetNextRecord(&qual,&qual_length);



      nextAlignment.Name = "idk";            //TODO
      nextAlignment.Length = base_length;
      nextAlignment.QueryBases.assign(base,base_length);
      nextAlignment.AlignedBases = "idk";             //TODO
      nextAlignment.Qualities.assign(qual,qual_length);
      nextAlignment.TagData = "idk";             //TODO
      nextAlignment.RefID = result->position().ref_index();
      nextAlignment.Position = result->position().position();
      nextAlignment.Bin = 1234;                 //TODO
      nextAlignment.MapQuality = result->mapping_quality();
      nextAlignment.AlignmentFlag = result->flag();
      nextAlignment.CigarData.clear();
      const char* cigar;
      size_t cigar_len;
      cigar = result->cigar().c_str();
      cigar_len = result->cigar().length();
      char op;
      int op_len;
      while(cigar_len > 0)
      {
        size_t len = parseNextOp(cigar, op, op_len);
        cigar += len;
        cigar_len -= len;
        BamTools::CigarOp ops(op,op_len);
        nextAlignment.CigarData.push_back(ops);
      }
      nextAlignment.MateRefID = result->next_position().ref_index();
      nextAlignment.MatePosition = result->next_position().ref_index();
      nextAlignment.InsertSize = result->template_length();
      nextAlignment.Filename = "idk";                //TODO

      Status s = top.second.first.first->GetNextResult(result_global[top.second.second]);
      if(s.ok()){
        multiReader.push(ResultPair(&result_global[top.second.second],ReaderPair(ResBaseQual(top.second.first.first,BaseQualPair(top.second.first.second.first,top.second.first.second.second)),top.second.second)));
      }
      else{
        Status s = DequeueElement(ctx,top.second.second);
      }
      multiReader.pop();
      return true;
    }
    else
      return false;
  }

  bool AGDMultiReader::getNextAlignment(SeqLib::BamRecord& nextRecord)
  {

    if(!multiReader.empty()){
      auto &top = multiReader.top();
      const char* base;
      size_t base_length;
      const char* qual;
      size_t qual_length;
      Alignment* result = top.first ;
      Status sb = top.second.first.second.first->GetNextRecord(&base,&base_length);
      Status sq = top.second.first.second.second->GetNextRecord(&qual,&qual_length);
      nextRecord.init();
      //cout << result->position().position()<<endl;
      nextRecord.SetMapQuality(result->mapping_quality());
      nextRecord.SetID(result->position().ref_index());
      nextRecord.SetChrID(result->position().ref_index());
      nextRecord.SetChrIDMate(result->next_position().ref_index());
      string base_str(base,base_length);
      nextRecord.SetSequence(base_str);
      nextRecord.SetPositionMate(result->next_position().position());
      string qual_str(qual,qual_length);
      cout << qual_str.length() << " "<< base_str.length()<<endl;
      nextRecord.SetQualities(qual_str,qual_length);

      nextRecord.SetPosition(result->position().position());
      nextRecord.SetQname(nextRecord.ChrName());
      SeqLib::Cigar cgr;
      const char* cigar;
      size_t cigar_len;
      cigar = result->cigar().c_str();
      cigar_len = result->cigar().length();
      char op;
      int op_len;
      while(cigar_len > 0)
      {
        size_t len = parseNextOp(cigar, op, op_len);
        cigar += len;
        cigar_len -= len;
        SeqLib::CigarField field(op,op_len);
        cgr.add(field);
      }
      nextRecord.SetCigar(cgr);




      Status s = top.second.first.first->GetNextResult(result_global[top.second.second]);
      if(s.ok()){
        multiReader.push(ResultPair(&result_global[top.second.second],ReaderPair(ResBaseQual(top.second.first.first,BaseQualPair(top.second.first.second.first,top.second.first.second.second)),top.second.second)));
      }
      else{
        Status s = DequeueElement(ctx,top.second.second);
      }
      multiReader.pop();
      return true;
    }
    else
      return false;
  }





  Status AGDMultiReader::Init(OpKernelContext *ctx,int i) {
    TF_RETURN_IF_ERROR(LookupResource(ctx, HandleFromInput(ctx, i), &input_queue_[i]));

    // auto &input_dtypes = input_queue_->component_dtypes();
    // if (!(input_dtypes.at(0) == DT_STRING)) {
    //   return Internal("Barrier: input or output queue has a non-string type for first element");
    // }
    return Status::OK();
  }


  Status AGDMultiReader::LoadDataResource(OpKernelContext* ctx, const Tensor* handle_t,ResourceContainer<Data>** container) {
    auto rmgr = ctx->resource_manager();
    auto handles_vec = handle_t->vec<string>();

    TF_RETURN_IF_ERROR(rmgr->Lookup(handles_vec(0), handles_vec(1), container));
    return Status::OK();
  }


  void AGDMultiReader::PQInit(OpKernelContext* ctx){
    for(int i=0;i<vec_size;i++){
      Status s = DequeueElement(ctx,i);
    }
  }


  Status AGDMultiReader::DequeueElement(OpKernelContext *ctx,int i) {
    Notification n;
    int sf = -1;
    //Status s;
    input_queue_[i]->TryDequeue(ctx, [&](const QueueInterface::Tuple &tuple) {
        //out << input_queue_->size();

      //cout << tuple.size() << endl;
      if(tuple.size()==0)
      {
        sf = 2;
        n.Notify();
      }
      else
      {

        base_t = &tuple[0];
        // auto &base = base_t.scalar<string>()();
        quality_t = &tuple[1];
        meta_t = &tuple[2];
        // auto &quality = quality_t.scalar<string>()();
        result_t = &tuple[3];
        // auto &result = result_t.scalar<string>()();
        num_records_t = &tuple[4];
        // auto &num_records = num_records_t.scalar<string>()();

        auto num_records = num_records_t->scalar<int32>()();
         ResourceContainer<Data>* bases_data, *qual_data,*meta_data, *result_data;
        //ResourceContainer<Data>* result_data;

        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, meta_t, &meta_data));
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, base_t, &bases_data));
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, quality_t, &qual_data));
        OP_REQUIRES_OK(ctx, LoadDataResource(ctx, result_t, &result_data));
        AGDRecordReader base_reader(bases_data, num_records);
        AGDRecordReader qual_reader(qual_data, num_records);
        AGDRecordReader meta_reader(meta_data,num_records);
        AGDResultReader results_reader(result_data, num_records);
        Alignment result;
        Status s = results_reader.GetNextResult(result);
        //cout << result.position().position()<<endl;
        if(results_reader_global.size()<=i)
        {
          results_reader_global.push_back(results_reader);
          base_reader_global.push_back(base_reader);
          qual_reader_global.push_back(qual_reader);
          result_global.push_back(result);
        }
        else
        {
          results_reader_global[i] = results_reader;
          base_reader_global[i] = base_reader;
          qual_reader_global[i] = qual_reader;
          result_global[i] = result;
        }

        if(s.ok())
          multiReader.push(ResultPair(&result_global[i],ReaderPair(ResBaseQual(&results_reader_global[i],BaseQualPair(&base_reader_global[i],&qual_reader_global[i])),i)));

        cout << ++chunk_count<< " from queue " << i <<  " put inside the multireader"<< endl;
        //print_alignment(&results_reader);
        //newBaseReader = &base_reader;
        //newQualReader = &qual_reader;
        //*newResultReader = results_reader;
        resource_releaser(bases_data);
        resource_releaser(qual_data);
        resource_releaser(meta_data);
        resource_releaser(result_data);



        n.Notify();
      }

    });
    n.WaitForNotification();
    if(sf==2)
    {
      //cout << "dsdsd";
      return NotFound("reached end of queue");
    }
    return Status :: OK();
  }






}

