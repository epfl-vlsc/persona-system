#include "dense_data.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  using namespace std;

  DenseReadData::~DenseReadData() {
    // TODO we'll probably still need something here
  }

  DenseReadData::DenseReadData(RecordBuffer bases,
                               RecordBuffer qualities,
                               RecordBuffer metadata) : bases_(bases),
                                                        qualities_(qualities),
                                                        metadata_(metadata)
  {
    auto b = bases->RecordCount();
    auto q = qualities->RecordCount();
    if (b != q) {
      LOG(ERROR) << "bases (" << b << ") and qualities (" << q << ") have different lengths";
    }

    if (metadata) {
      auto m = metadata->RecordCount();
      if (m != b) {
        LOG(ERROR) << "metadata (" << m << ") and bases (" << b << ") have different lengths";
      }
    }
  }

  Status DenseReadData::set_metadata(RecordBuffer metadata) {
    auto md_record_count = metadata->RecordCount();
    auto base_record_count = bases_->RecordCount();
    if (base_record_count != md_record_count) {
      return errors::InvalidArgument("Metadata record count (", md_record_count, ") is not the same as base record count (", base_record_count, ")");
    }
    metadata_ = metadata;
    return Status::OK();
  }

  bool DenseReadData::has_metadata() {
    return metadata_ != nullptr;
  }

  size_t DenseReadData::num_records() {
    return bases_->RecordCount();
  }

  Status DenseReadData::qualities(size_t index, const char **data, size_t *length) {
    return qualities_->GetRecordAtIndex(index, data, length);
  }

  Status DenseReadData::bases(size_t index, const char **data, size_t *length) {
    return bases_->GetRecordAtIndex(index, data, length);
  }

  Status DenseReadData::metadata(size_t index, const char **data, size_t *length) {
    if (has_metadata()) {
      return metadata_->GetRecordAtIndex(index, data, length);
    }

    return errors::NotFound("Accessing metadata on Dense Record that has none");
  }

  Status DenseReadData::get_next_record(const char **bases, size_t *bases_length,
                                        const char **qualities, size_t *qualities_length) {
    using namespace errors;

    if (exhausted()) {
      return ResourceExhausted("next record unavailable for get_next_record");
    }

    auto a = bases_->GetNextRecord(bases, bases_length);
    if (a.ok()) {
      a = qualities_->GetNextRecord(qualities, qualities_length);
      if (a.ok()) {
        iter_++;
      }
    }

    return a;
  }

  Status DenseReadData::get_next_record(const char **bases, size_t *bases_length,
                                        const char **qualities, size_t *qualities_length,
                                        const char **metadata, size_t *metadata_length) {
    using namespace errors;
    if (!has_metadata()) {
      return Internal("Attempting to get_next_record on a record without metadata");
    }

    // This call will advance iter_. Don't worry about resetting if it errors
    auto a = get_next_record(bases, bases_length, qualities, qualities_length);
    if (a.ok()) {
      a = metadata_->GetNextRecord(metadata, metadata_length);
    }

    return a;
  }

} // namespace tensorflow {
