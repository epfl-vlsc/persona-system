#include "dense_reads.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;

  DenseReadResource::DenseReadResource(size_t num_records, DataContainer *bases, DataContainer *quals, DataContainer *meta) :
    bases_(bases), quals_(quals), meta_(meta), num_records_(num_records)
  {
    auto idx_offset = num_records * sizeof(RecordTable::IndexValue);
    auto b = bases->get()->data();
    base_idx_ = reinterpret_cast<const RecordTable*>(b);
    base_data_ = b + idx_offset;

    if (quals) {
      auto q = quals->get()->data();
      qual_idx_ = reinterpret_cast<const RecordTable*>(q);
      qual_data_ = q + idx_offset;
    } else {
      qual_idx_ = nullptr;
      qual_data_ = nullptr;
    }

    if (meta) {
      auto m = meta->get()->data();
      meta_idx_ = reinterpret_cast<const RecordTable*>(m);
      meta_data_ = m + idx_offset;
    } else {
      meta_idx_ = nullptr;
      meta_data_ = nullptr;
    }
  }

  DenseReadResource::~DenseReadResource()
  {
    if (bases_)
      bases_->release();
    if (quals_)
      quals_->release();
    if (meta_)
      meta_->release();
  }

  bool DenseReadResource::reset_iter()
  {
    record_idx_ = 0;
    auto idx_offset = num_records_ * sizeof(RecordTable::IndexValue);
    base_data_ = bases_->get()->data() + idx_offset;
    qual_data_ = quals_->get()->data() + idx_offset;
    meta_data_ = meta_->get()->data() + idx_offset;

    return true;
  }


  bool DenseReadResource::has_qualities()
  {
    return quals_ != nullptr;
  }

  bool DenseReadResource::has_metadata()
  {
    return meta_ != nullptr;
  }

  Status DenseReadResource::get_next_record(const char **bases, std::size_t *bases_length,
                                            const char **qualities, std::size_t *qualities_length,
                                            const char **metadata, std::size_t *metadata_length)
  {
    if (record_idx_ < num_records_) {
      auto base_len = base_idx_->relative_index[record_idx_];
      *bases_length = base_len;
      *bases = base_data_;
      base_data_  += base_len;

      if (quals_) {
        auto qual_len = qual_idx_->relative_index[record_idx_];
        *qualities_length = qual_len;
        *qualities = qual_data_;
        qual_data_ += qual_len;
      } else {
        *qualities = nullptr;
        *qualities_length = 0;
      }

      if (meta_) {
        auto meta_len = meta_idx_->relative_index[record_idx_];
        *metadata_length = meta_len;
        *metadata = meta_data_;
        meta_data_ += meta_len;
      } else {
        *metadata = nullptr;
        *metadata_length = 0;
      }

      record_idx_++;
    } else {
      return ResourceExhausted("dense record container exhausted");
    }
  }

} // namespace tensorflow {
