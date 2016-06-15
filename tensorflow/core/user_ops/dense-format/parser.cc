#include "parser.h"
#include <utility>
#include "decompress.h"
#include "util.h"

namespace tensorflow {

  using namespace std;

  RecordParser::RecordParser(std::size_t size)
  {
    buffer_.reserve(size);
  }

  Status RecordParser::ParseNew(const char* data, const std::size_t length, const bool verify, vector<char> &scratch, vector<char> &index_scratch)
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
    string type_string;
    switch (record_type) {
    default:
      return Internal("Invalid record type", file_header->record_type);
    case RecordType::BASES:
      type_string = "bases";
      break;
    case RecordType::QUALITIES:
      type_string = "qualities";
      break;
    case RecordType::COMMENTS:
      type_string = "metadata";
      break;
    }

    auto payload_start = data + file_header->segment_start;
    auto payload_size = length - file_header->segment_start;

    Status status;
    auto compression_type = static_cast<CompressionType>(file_header->compression_type);
    switch (compression_type) {
    case CompressionType::GZIP:
      status = decompressGZIP(payload_start, payload_size, buffer_);
      break;
    case CompressionType::UNCOMPRESSED:
      status = copySegment(payload_start, payload_size, buffer_);
      break;
    default:
      status = errors::InvalidArgument("Compressed type '", file_header->compression_type, "' doesn't match to any valid or supported compression enum type");
      break;
    }
    TF_RETURN_IF_ERROR(status);

    const size_t index_size = file_header->last_ordinal - file_header->first_ordinal;
    if (buffer_.size() < index_size * 2) {
      return Internal("FillBuffer: expected at least ", index_size*2, " bytes, but only have ", buffer_.size());
    }

    records = reinterpret_cast<const RecordTable*>(buffer_.data());

    if (verify) {
      size_t data_size = 0;
      // This iteration is expensive, which is why this is optional. Run with perf to get an idea
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
    }

    if (static_cast<RecordType>(file_header->record_type) == RecordType::BASES) {
      scratch.clear(); index_scratch.clear();

      uint8_t current_record_length;
      const char* start_ptr = &buffer_[index_size];
      const BinaryBaseRecord *bases;

      for (uint64_t i = 0; i < index_size; ++i) {
        current_record_length = records->relative_index[i];
        bases = reinterpret_cast<const BinaryBaseRecord*>(start_ptr);
        start_ptr += current_record_length;

        TF_RETURN_IF_ERROR(bases->appendToVector(current_record_length, scratch, index_scratch));
      }

      // append everything in converted_records to the index
      buffer_.clear();
      buffer_.reserve(index_scratch.size() + scratch.size());
      TF_RETURN_IF_ERROR(copySegment(&index_scratch[0], index_scratch.size(), buffer_));
      TF_RETURN_IF_ERROR(appendSegment(&scratch[0], scratch.size(), buffer_));
      current_offset_ = index_scratch.size(); // FIXME I think this is wrong
    } else {
      current_offset_ = index_size;
    }

    total_records_ = index_size;
    file_header_ = *file_header;
    current_record_ = 0;
    return Status::OK();
  }

  void RecordParser::reset() {
    ResetIterator();
    buffer_.clear();
    total_records_ = 0;
    current_offset_ = 0;
    valid_record_ = false;
  }

  void RecordParser::ResetIterator()
  {
    current_record_ = 0;
  }

  size_t RecordParser::RecordCount()
  {
    return total_records_;
  }

  bool RecordParser::HasNextRecord()
  {
    return current_record_ < total_records_;
  }

  Status RecordParser::GetNextRecord(const char** value, size_t *length)
  {
    using namespace errors;
    using namespace format;

    if (!HasNextRecord()) {
      return ResourceExhausted("No more next records available in RecordParser");
    }

    auto current_record_length = records->relative_index[current_record_];
    auto start_ptr = &buffer_[current_offset_];
    *value = start_ptr;
    *length = current_record_length;

    current_record_++;
    current_offset_ += current_record_length;

    return Status::OK();
  }

  Status RecordParser::GetRecordAtIndex(size_t index, const char **value, size_t *length)
  {
    using namespace errors;
    using namespace format;

    if (index >= total_records_ || index < 0) {
      return Internal("Record access attempt at index ", index, ", with only ", total_records_, " records");
    }

    size_t i = 0;
    for (size_t j = 0; j < index; j++) {
      i += records->relative_index[j];
    }

    auto record_ptr = &buffer_[i];
    auto record_len = records->relative_index[index];
    *value = record_ptr;
    *length = record_len;
    return Status::OK();
  }

} // namespace tensorflow {
