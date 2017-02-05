#include "parser.h"
#include <utility>
#include <array>
#include "compression.h"
#include "util.h"
#include "format.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/contrib/persona/kernels/lttng/tracepoints.h"
#include <cstdint>

namespace tensorflow {

  namespace {
    unsigned char nst_nt4_table[256] = {
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
      4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
    };
  }

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

  Status RecordParser::ParseNew(const char* data, const std::size_t length, const bool verify, Buffer *result_buffer, uint64_t *first_ordinal, uint32_t *num_records, bool unpack, bool twobit)
  {
    using namespace errors;
    using namespace format;
    reset();

    if (length < sizeof(FileHeader)) {
      return Internal("AGDReader::FillBuffer: needed min length ", sizeof(FileHeader),
                      ", but only received ", length);
    }
    auto file_header = reinterpret_cast<const FileHeader*>(data);
    auto record_type = static_cast<RecordType>(file_header->record_type);
    switch (record_type) {
    case RecordType::BASES:
    case RecordType::QUALITIES:
    case RecordType::ALIGNMENT:
    case RecordType::COMMENTS:
      break;
    default:
      return Internal("Invalid record type ", file_header->record_type);
    }

    auto payload_start = data + file_header->segment_start;
    auto payload_size = length - file_header->segment_start;

    Status status;
    auto compression_type = static_cast<CompressionType>(file_header->compression_type);
    switch (compression_type) {
    case CompressionType::GZIP:
      status = decompressGZIP(payload_start, payload_size, result_buffer);
      break;
    case CompressionType::UNCOMPRESSED:
      status = result_buffer->WriteBuffer(payload_start, payload_size);
      break;
    default:
      status = errors::InvalidArgument("Compressed type '", file_header->compression_type, "' doesn't match to any valid or supported compression enum type");
      break;
    }
    const size_t index_size = file_header->last_ordinal - file_header->first_ordinal;
    TF_RETURN_IF_ERROR(status);

    if (result_buffer->size() < index_size * 2) {
      return Internal("FillBuffer: expected at least ", index_size*2, " bytes, but only have ", result_buffer->size());
    }

    records = reinterpret_cast<const RelativeIndex*>(result_buffer->data());

    if (verify) {
      size_t data_size = 0;
      // This iteration is expensive, which is why this is optional. Run with perf to get an idea
      for (uint64_t i = 0; i < index_size; ++i) {
        data_size += records[i];
      }

      const size_t expected_size = result_buffer->size() - index_size;
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
      if (unpack) {
        conversion_scratch_.reset(); index_scratch_.reset();

        uint8_t current_record_length;
        const char* start_ptr = &(*result_buffer)[index_size];
        const BinaryBases *bases;

        for (uint64_t i = 0; i < index_size; ++i) {
          current_record_length = records[i];
          bases = reinterpret_cast<const BinaryBases*>(start_ptr);
          start_ptr += current_record_length;

          TF_RETURN_IF_ERROR(append(bases, current_record_length, conversion_scratch_, index_scratch_));
        }

        if (twobit) {
          LOG(INFO) << "calling two bit with unpack";
          for (int i = 0; i < conversion_scratch_.size(); ++i) {// convert to 2-bit encoding 
            //LOG(INFO) << "converting " << conversion_scratch_[i] << " to " << (int)nst_nt4_table[(int)conversion_scratch_[i]];
            conversion_scratch_[i] = nst_nt4_table[(int)conversion_scratch_[i]];
            //LOG(INFO) << "is now: " << (int)(conversion_scratch_[i]);
            if (conversion_scratch_[i] > 4) return Internal("sequence is fucked");
          }
        }
        // append everything in converted_records to the index
        result_buffer->reserve(index_scratch_.size() + conversion_scratch_.size());
        TF_RETURN_IF_ERROR(result_buffer->WriteBuffer(&index_scratch_[0], index_scratch_.size()));
        TF_RETURN_IF_ERROR(result_buffer->AppendBuffer(&conversion_scratch_[0], conversion_scratch_.size()));
      } else if (twobit) {
        LOG(INFO) << "calling two bit without unpack";
        const char* start_ptr = &(*result_buffer)[index_size];
        conversion_scratch_.resize(payload_size - index_size);
        for (int i = 0; i < payload_size-index_size; ++i) // convert to 2-bit encoding
          conversion_scratch_[i] = nst_nt4_table[(int)start_ptr[i]];

        TF_RETURN_IF_ERROR(result_buffer->WriteBuffer(payload_start, index_size));
        TF_RETURN_IF_ERROR(result_buffer->AppendBuffer(&conversion_scratch_[0], conversion_scratch_.size()));
      }
    }

    *first_ordinal = file_header->first_ordinal;
    *num_records = index_size;
    return Status::OK();
  }

  void RecordParser::reset() {
    conversion_scratch_.reset();
    index_scratch_.reset();
  }

  RecordParser::RecordParser() {
    init_table();
  }

} // namespace tensorflow {
