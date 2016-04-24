// Sam Whitlock (sam.whitlock@epfl.ch)

#include "reader_async_base.h"

namespace tensorflow {

ReaderAsyncBase::InputChunk::GetChunk(const void** data, std::size_t *length)
{
  *data = data_;
  *length = length_;
}

ReaderAsyncBase::ReaderAsyncBase() :
{}

Status ReaderAsyncBase::EnqueueNextChunk(InputChunk &&input_chunk)
{
  
}

Status ReaderAsyncBase::GetNextWork(InputChunk *next_chunk)
{
  
}

void ReaderAsyncBase::Read(QueueInterface* queue, string* key, string* value,
          OpKernelContext* context) override
{

}

void ReaderAsyncBase::ReadBatch(QueueInterface* queue, 
               std::function<string*(int)> batch_loader, 
               int batch_size, string* key, OpKernelContext* context,
               int* produced) override
{

}

Status ReaderAsyncBase::Reset() override
{

}

int64 ReaderAsyncBase::NumRecordsProduced() override
{

}

int64 ReaderAsyncBase::NumWorkUnitsCompleted() override
{

}

Status ReaderAsyncBase::SerializeState(string* state) override
{

}

Status ReaderAsyncBase::RestoreState(const string& state) override
{

}

} // namespace tensorflow {
