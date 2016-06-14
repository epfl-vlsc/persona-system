#include "read_data.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  using namespace std;

  ReadData::~ReadData() {};

  bool ReadData::has_metadata() {
    return false;
  }

  bool ReadData::exhausted() {
    return !(iter_ < num_records());
  }

  Status ReadData::metadata(size_t index, const char **data, size_t *length) {
    return errors::Unimplemented("Attempting to access metadata from an unsupporting class");
  }

  Status ReadData::get_next_record(const char **bases, size_t *bases_length,
                         const char **qualities, size_t *qualities_length) {
    auto a = get_current_record(bases, bases_length, qualities, qualities_length);
    if (a.ok()) {
      ++iter_;
    }
    return a;
  }

  Status ReadData::get_next_record(const char **base, size_t *base_length,
                         const char **quality, size_t *quality_length,
                         const char **metadata_record, size_t *metadata_record_length) {
    if (has_metadata()) {
      auto a = get_current_record(base, base_length, quality, quality_length);
      if (a.ok()) {
        a = metadata(iter_, metadata_record, metadata_record_length);
        if (a.ok()) {
          iter_++;
        }
      }
      return a;
    }
    return errors::Unimplemented("Attempting to get metadata-style record from record type that doesn't support it");
  }

  Status ReadData::get_current_record(const char **base_record, size_t *base_record_length,
                                    const char **quals, size_t *qualities_length) {
    if (iter_ < num_records()) {
      auto a = bases(iter_, base_record, base_record_length);
      if (a.ok()) {
        a = qualities(iter_, quals, qualities_length);
      }
      return a;
    }

    return errors::ResourceExhausted("unable to get next record at iter ", iter_);
  }

  void ReadData::reset_iter() {
    iter_ = 0;
  }

  string ReadData::DebugString() {
    return "a ReadData data reader";
  }
} // namespace tensorflow {
