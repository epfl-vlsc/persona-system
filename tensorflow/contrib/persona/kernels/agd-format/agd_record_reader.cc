#include "agd_record_reader.h"

namespace tensorflow {

  using namespace errors;
  using namespace std;
  using namespace format;

  // TODO we should probably do some more checks on this
  AGDRecordReader::AGDRecordReader(ResourceContainer<Data>* resource, size_t num_records) :
    num_records_(num_records) {
    auto idx_offset = num_records * sizeof(RelativeIndex);
    auto base_data = resource->get()->data();
    index_ = reinterpret_cast<const RelativeIndex*>(base_data);
    cur_data_ = data_ = base_data + idx_offset;
    InitializeIndex();
  }
    
  AGDRecordReader::AGDRecordReader(const char* resource, size_t num_records) :
    num_records_(num_records) {
    auto idx_offset = num_records * sizeof(RelativeIndex);
    auto base_data = resource;
    index_ = reinterpret_cast<const RelativeIndex*>(base_data);
    cur_data_ = data_ = base_data + idx_offset;
    InitializeIndex();
  }

  AGDRecordReader AGDRecordReader::fromUncompressed(ResourceContainer<Data>* resource, bool *success) {
    const char* a = nullptr;
    auto d = resource->get();
    auto d_sz = d->size();
    auto d_data = d->data();
    if (d_sz < sizeof(FileHeader)) {
      LOG(ERROR) << "Received a chunk with less than " << sizeof(FileHeader) << " bytes needed for the header";
      *success = false;
      return AGDRecordReader(a, 0);
    }

    auto header = reinterpret_cast<const FileHeader*>(d_data);
    int64_t num_records = header->last_ordinal - header->first_ordinal;
    if (num_records < 1) {
      LOG(ERROR) << "Receive a chunk with " << num_records << " records";
      *success = false;
      return AGDRecordReader(a, 0);
    }

    size_t minimum_index_size = (d_sz - sizeof(FileHeader)) * sizeof(RelativeIndex);
    if (minimum_index_size < num_records) {
      LOG(ERROR) << "Received invalid chunk. Header specifices " << num_records << " records, but payload is only " << minimum_index_size << " bytes";
      *success = false;
      return AGDRecordReader(a, 0);
    }

    *success = true;
    return AGDRecordReader(d_data + sizeof(FileHeader), num_records);
  }

  void AGDRecordReader::InitializeIndex() {
    absolute_index_.clear();
    size_t current = 0;
    for (size_t i = 0; i < num_records_; ++i) {
      absolute_index_.push_back(current);
      current += index_[i];
    }
  }

  void AGDRecordReader::Reset() {
    cur_data_ = data_;
    cur_record_ = 0;
  }

  Status AGDRecordReader::PeekNextRecord(const char** data, size_t* size) {
    if (cur_record_ < num_records_) {
      *size = (size_t) index_[cur_record_];
      *data = cur_data_;
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
    return Status::OK();
  }

  Status AGDRecordReader::GetNextRecord(const char** data, size_t* size) {
    auto s = PeekNextRecord(data, size);
    if (s.ok()) {
      cur_data_ += index_[cur_record_++];
    }
    return s;
  }

  Status AGDRecordReader::GetRecordAt(size_t index, const char** data, size_t* size) {
    if (index < num_records_) {
      *size = (size_t) index_[index];
      *data = data_ + absolute_index_[index];
    } else {
      return OutOfRange("agd record random access out of range");
    }
    return Status::OK();
  }


  AGDRemoteRecordReader::AGDRemoteRecordReader(string filename, size_t num_records, 
        char* buffer, uint64_t buffer_size, librados::IoCtx* io_ctx) :
    io_ctx_(io_ctx), base_buf_(buffer), base_size_(buffer_size), num_records_(num_records),
    filename_(filename) {

    LOG(INFO) << "i got base pointer: " << (uint64_t)buffer;
    LOG(INFO) << "with base size: " << base_size_;
    current_offset_ = sizeof(format::FileHeader);
    buf_0_.data = base_buf_ + num_records_;
    auto size = (buffer_size - num_records_) / 2;
    buf_1_.data = buf_0_.data + size;
    buf_0_.size = base_size_ - (base_buf_ + base_size_ - buf_1_.data) - num_records_;
    buf_1_.size = base_size_ - buf_0_.size - num_records_;
    LOG(INFO) << "buf 0 size: " << buf_0_.size;
    LOG(INFO) << "buf 1 size: " << buf_1_.size;
    buf_0_.recs = 0;
    buf_1_.recs = 0;
  }

  Status AGDRemoteRecordReader::ReadData(char* dest, uint64_t size) {
    librados::bufferlist read_buf;
    read_buf.push_back(ceph::buffer::create_static(size, dest));
    librados::AioCompletion *read_completion = librados::Rados::aio_create_completion();
    LOG(INFO) << "executing read of " << size << " bytes";
    int ret = io_ctx_->aio_read(filename_, read_completion, &read_buf, size, current_offset_);
    if (ret < 0) {
      return errors::Internal("Couldn't start read object in remote record reader");
    }
    read_completion->wait_for_complete();
    ret = read_completion->get_return_value();
    if (ret < 0) {
      LOG(INFO) << "Couldn't read object! error " << ret;
      return errors::Internal("Couldn't read ceph object in remote record reader");
    }
    read_buf.clear();
    read_completion->release();

    current_offset_ += size;
    return Status::OK();
  }

  // read in the index, and fill the first data buffer full of records
  // as much as will fit
  Status AGDRemoteRecordReader::Initialize() {
    if (init)
      return Status::OK();

    Status s = ReadData(base_buf_, num_records_);
    if (!s.ok())
      return s;
    index_ = reinterpret_cast<const RelativeIndex*>(base_buf_);
    uint64_t total = 0;
    size_t i;
    for (i = 0; i < num_records_; i++) {
      if (total + index_[i] < buf_0_.size)
        total += index_[i];
      else
        break;
    }
    cur_record_prefetched_ = i;
    s = ReadData(buf_0_.data, total);
    if (!s.ok())
      return s;
    buf_0_.recs = i;
    buf_0_.cur_data = buf_0_.data;
    active_buf_ = &buf_0_;
    init = true;
    return Status::OK();
  }

  AGDRemoteRecordReader::RecordBuffer* AGDRemoteRecordReader::OtherBuffer() {
    if (active_buf_ == &buf_0_)
      return &buf_1_;
    else
      return &buf_0_;
  }

  Status AGDRemoteRecordReader::GetNextRecord(const char** data, size_t* size) {
    auto s = PeekNextRecord(data, size);
    if (s.ok()) {
      active_buf_->cur_data += index_[cur_record_++];
      if (active_buf_->recs > 1)
        active_buf_->recs--;
      else if (active_buf_->recs == 1) {
        // we need to switch buffers, but make sure the other is filled first
        auto other_buf = OtherBuffer();
        if (other_buf->recs == 0 && cur_record_prefetched_ < num_records_) {
          mutex_lock l(mu_);
          // may need to make recs volatile?
          LOG(INFO) << "GetNextRecord is waiting for more data ...";
          ready_cv_.wait(l, [other_buf]() {
              return other_buf->recs == 0;
            });
        }
        active_buf_->recs = 0;
        active_buf_ = other_buf;
        return s;
      }
    }
    return s;
  }
  
  Status AGDRemoteRecordReader::PeekNextRecord(const char** data, size_t* size) {
    if (!init)
      return Internal("Tried to use uninitialized remote record reader");

    if (cur_record_ < num_records_) {
      *size = (size_t) index_[cur_record_];
      //*data = cur_data_;
      if (active_buf_->recs == 0)
        return Internal("The active buffer of a remote record reader has 0 records!");

      *data = active_buf_->cur_data;
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
    return Status::OK();
  }

  Status AGDRemoteRecordReader::PrefetchRecords() {
    if (!init)
      return Internal("Tried to use uninitialized remote record reader");

    auto other_buf = OtherBuffer();
    if (other_buf->recs == 0 && cur_record_prefetched_ < num_records_) { 
      // fill that shit
      uint64_t total = 0;
      size_t i;
      for (i = cur_record_prefetched_; i < num_records_; i++) {
        if (total + index_[i] < other_buf->size) {
          total += index_[i]; 
        } else
          break;
      }
      auto recs = i - cur_record_prefetched_;
      if (recs == 0) 
        return Internal("No records were prefetched, you may need to increase the file buffer size");
      cur_record_prefetched_ = i;
      Status s = ReadData(other_buf->data, total);
      if (!s.ok())
        return s;
      other_buf->cur_data = other_buf->data;
      other_buf->recs = recs;
      ready_cv_.notify_one();
    } else if (cur_record_prefetched_ == num_records_)
      return ResourceExhausted("Remote data in ceph is fully prefetched");

    return Status::OK();
  }

} // namespace tensorflow {
