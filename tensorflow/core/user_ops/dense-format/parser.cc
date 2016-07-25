#include "parser.h"
#include <utility>
#include <array>
#include "compression.h"
#include "util.h"
#include "format.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/user_ops/lttng/tracepoints.h"
#include <cstdint>

namespace tensorflow {

  using namespace std;
  namespace {
    volatile bool table_needs_init_ = true;
    const size_t base_width = 3;
    const size_t num_bases = 512; // 2*(enum_bit_width{3} * base_width{3})
    const auto mask = ~(~0 << format::BinaryBases::base_width);
    array<BaseMapping<base_width>, num_bases> base_table_;
    mutex base_table_mu_;

    template <size_t N>
    BaseMapping<N>
    make_result(size_t key) {
      using namespace format;
      array<char, N> ret;
      ret.fill('\0');
      uint64_t masked;
      bool run = true, valid = true;
      char c;
      size_t usable_characters = 0;
      for (size_t i = 0; run && i < ret.size(); ++i) {
        masked = static_cast<BaseAlphabet>(key & mask);
        switch (masked) {
        default:
          // if a bad value is found, we need to set to a default "bad" value, as below
          run = false;
          valid = false;
          break;
        case BaseAlphabet::A:
          c = 'A';
          break;
        case BaseAlphabet::T:
          c = 'T';
          break;
        case BaseAlphabet::C:
          c = 'C';
          break;
        case BaseAlphabet::G:
          c = 'G';
          break;
        case BaseAlphabet::N:
          c = 'N';
          break;
        case BaseAlphabet::END:
          run = false;
          break;
        }
        if (run) {
          ret[i] = c;
          usable_characters++;
          key >>= BinaryBases::base_width;
        }
      }

      if (!valid) {
        ret.fill('\0');
        usable_characters = 0;
      }

      return BaseMapping<N>(ret, usable_characters);
    }

    void init_table_locked() {
      for (size_t i = 0; i < base_table_.size(); ++i) {
        base_table_[i] = make_result<base_width>(i);
      }
    }

    void init_table() {
      if (table_needs_init_) {
        mutex_lock l(base_table_mu_);
        if (table_needs_init_) {
          init_table_locked();
          table_needs_init_ = false;
        }
      }
    }
  }

  const BaseMapping<3>*
  lookup_triple(const size_t bases) {
    auto b = bases & 0x1ff;
    const auto &a = base_table_[b];
    if (a.get()[0] == 'Z') {
      return nullptr;
    }
    return &a;
  }

  Status RecordParser::ParseNew(const char* data, const std::size_t length, const bool verify, vector<char> &result, uint64_t *first_ordinal, uint32_t *num_records)
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
    case RecordType::BASES:
    case RecordType::QUALITIES:
    case RecordType::COMMENTS:
      break;
    default:
      return Internal("Invalid record type", file_header->record_type);
    }

    auto payload_start = data + file_header->segment_start;
    auto payload_size = length - file_header->segment_start;

    Status status;
    auto compression_type = static_cast<CompressionType>(file_header->compression_type);
    auto start = clock();
    switch (compression_type) {
    case CompressionType::GZIP:
      status = decompressGZIP(payload_start, payload_size, result);
      break;
    case CompressionType::UNCOMPRESSED:
      status = copySegment(payload_start, payload_size, result);
      break;
    default:
      status = errors::InvalidArgument("Compressed type '", file_header->compression_type, "' doesn't match to any valid or supported compression enum type");
      break;
    }
    tracepoint(bioflow, decompression, clock() - start);
    TF_RETURN_IF_ERROR(status);

    const size_t index_size = file_header->last_ordinal - file_header->first_ordinal;
    if (result.size() < index_size * 2) {
      return Internal("FillBuffer: expected at least ", index_size*2, " bytes, but only have ", result.size());
    }

    records = reinterpret_cast<const RecordTable*>(result.data());

    if (verify) {
      size_t data_size = 0;
      // This iteration is expensive, which is why this is optional. Run with perf to get an idea
      for (uint64_t i = 0; i < index_size; ++i) {
        data_size += records->relative_index[i];
      }

      const size_t expected_size = result.size() - index_size;
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
      start = clock();
      conversion_scratch_.clear(); index_scratch_.clear();

      uint8_t current_record_length;
      const char* start_ptr = &result[index_size];
      const BinaryBaseRecord *bases;

      for (uint64_t i = 0; i < index_size; ++i) {
        current_record_length = records->relative_index[i];
        bases = reinterpret_cast<const BinaryBaseRecord*>(start_ptr);
        start_ptr += current_record_length;

        TF_RETURN_IF_ERROR(bases->appendToVector(current_record_length, conversion_scratch_, index_scratch_));
      }

      // append everything in converted_records to the index
      result.clear();
      result.reserve(index_scratch_.size() + conversion_scratch_.size());
      TF_RETURN_IF_ERROR(copySegment(&index_scratch_[0], index_scratch_.size(), result));
      TF_RETURN_IF_ERROR(appendSegment(&conversion_scratch_[0], conversion_scratch_.size(), result));
      tracepoint(bioflow, base_conversion, clock() - start);
    }

    *first_ordinal = file_header->first_ordinal;
    *num_records = index_size;
    return Status::OK();
  }

  void RecordParser::reset() {
    conversion_scratch_.clear();
    index_scratch_.clear();
  }

  RecordParser::RecordParser() {
    init_table();
  }

} // namespace tensorflow {
