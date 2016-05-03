#include "parser.h"
#include "decompress.h"

namespace tensorflow {

  using namespace std;

  Status RecordParser::ParseNew(const char* data, const std::size_t length)
  {
    using namespace errors;
    using namespace format;

    reset();

    if (length < sizeof(FileHeader)) {
      return Internal("DenseReader::FillBuffer: needed min length ", sizeof(FileHeader),
                      ", but only received ", length);
    }
    auto file_header = reinterpret_cast<const FileHeader*>(data);
    auto record_type = static_cast<RecordType>(file_header->record_type);
    switch (record_type) {
    default:
      return Internal("Invalid record type", file_header->record_type);
    case RecordType::BASES:
    case RecordType::QUALITIES:
    case RecordType::COMMENTS:
      break;
    }

    auto payload_start = data + file_header->segment_start;
    auto payload_size = length - file_header->segment_start;

    Status status;
    if (static_cast<CompressionType>(file_header->compression_type) == CompressionType::BZIP2) {
      status = decompressBZIP2(payload_start, payload_size, buffer_);
    } else {
      status = copySegment(payload_start, payload_size, buffer_);
    }
    TF_RETURN_IF_ERROR(status);

    const size_t index_size = file_header->last_ordinal - file_header->first_ordinal;
    if (buffer_.size() < index_size * 2) {
      return Internal("FillBuffer: expected at least ", index_size*2, " bytes, but only have ", buffer_.size());
    } /* else if (index_size > batch_size_) {
      return Internal("FillBuffer: decompressed a chunk with ", index_size, " elements, but maximum batch size is ", batch_size_);
      } */

    records = reinterpret_cast<const RecordTable*>(buffer_.data());
    size_t data_size = 0;
    for (uint64_t i = 0; i < index_size; ++i) {
      data_size += records->relative_index[i];
    }

    const size_t expected_size = buffer_.size() - index_size;
    if (data_size != expected_size) {
      if (data_size < expected_size) {
        return OutOfRange("Expected a file size of ", expected_size, " bytes, but only found ",
                                    data_size, " bytes");
        } else {
          return OutOfRange("Expected a file size of ", expected_size, " bytes, but only found ",
                                    data_size, " bytes");
        }
      }

    total_records_ = index_size;
    current_record_ = 0;
    current_offset_ = index_size;
    file_header_ = *file_header;

    return Status::OK();
  }

  void RecordParser::reset()
  {
    buffer_.clear();
  }

  size_t RecordParser::RecordCount()
  {
    return total_records_;
  }

  bool RecordParser::HasNextRecord()
  {
    return current_record_ < total_records_;
  }

  Status RecordParser::GetNextRecord(string *value)
  {
    using namespace errors;
    using namespace format;

    if (!HasNextRecord()) {
      return ResourceExhausted("No more next records available in RecordParser");
    }

    auto current_record_length = records->relative_index[current_record_];
    auto start_ptr = &buffer_[current_offset_];
    if (file_header_.record_type == RecordType::BASES) {
      auto bases = reinterpret_cast<const format::BinaryBaseRecord*>(start_ptr);
      TF_RETURN_IF_ERROR(bases->toString(current_record_length, value));
    } else {
      *value = string(&buffer_[current_offset_], current_record_length);
    }
    current_record_++;
    current_offset_ += current_record_length;
    return Status::OK();
  }

} // namespace tensorflow {
