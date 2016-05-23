#include "data.h"

namespace tensorflow {
  using namespace std;

  ReadData::~ReadData() {};

  bool ReadData::has_metadata() {
    return false;
  }

  const char* ReadData::metadata(size_t index) {
    return nullptr;
  }

  size_t ReadData::metadata_length(size_t index) {
    return 0;
  }

  bool ReadData::get_next_record(const char **bases, size_t *bases_length,
                                 const char **qualities) {
    auto a = get_current_record(bases, bases_length, qualities);
    if (a) {
      ++iter_;
    }
    return a;
  }

  bool ReadData::get_next_record(const char **base_record, size_t *base_record_length,
                                 const char **quals,
                                 const char **metadata_record, size_t *metadata_record_length) {
    if (has_metadata()) {
      auto a = get_current_record(base_record, base_record_length, quals);
      if (a) {
        *metadata_record = metadata(iter_);
        *metadata_record_length = metadata_length(iter_++);
        return true;
      }
    }
    return false;
  }

  bool ReadData::get_current_record(const char **base_record, size_t *base_record_length,
                                    const char **quals) {
    if (iter_ < num_records()) {
      *base_record = bases(iter_);
      *base_record_length = bases_length(iter_);
      *quals = qualities(iter_);
      return true;
    }
    return false;
  }

  void ReadData::reset_iter() {
    iter_ = 0;
  }
} // namespace tensorflow {
