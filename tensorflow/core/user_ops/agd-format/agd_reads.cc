#include "dense_reads.h"

#include <utility>

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;
  using namespace format;

  AGDReadResource::AGDReadResource(size_t num_records, DataContainer *bases, DataContainer *quals, DataContainer *meta) :
    bases_(bases), quals_(quals), meta_(meta), num_records_(num_records)
  {
    // TODO probably chain this with the other constructor
    auto idx_offset = num_records * sizeof(RecordTable::IndexValue);
    auto b = bases->get()->data();
    base_idx_ = reinterpret_cast<const RecordTable*>(b);
    base_data_ = b + idx_offset;

    auto q = quals->get()->data();
    qual_idx_ = reinterpret_cast<const RecordTable*>(q);
    qual_data_ = q + idx_offset;

    auto m = meta->get()->data();
    meta_idx_ = reinterpret_cast<const RecordTable*>(m);
    meta_data_ = m + idx_offset;
  }

  AGDReadResource::AGDReadResource(size_t num_records, DataContainer *bases, DataContainer *quals) : bases_(bases), quals_(quals), num_records_(num_records)
  {
    auto idx_offset = num_records * sizeof(RecordTable::IndexValue);
    auto b = bases->get()->data();
    base_idx_ = reinterpret_cast<const RecordTable*>(b);
    base_data_ = b + idx_offset;

    auto q = quals->get()->data();
    qual_idx_ = reinterpret_cast<const RecordTable*>(q);
    qual_data_ = q + idx_offset;
  }

  AGDReadResource&
  AGDReadResource::operator=(AGDReadResource &&other)
  {
    bases_ = other.bases_;
    quals_ = other.quals_;
    meta_ = other.meta_;

    base_idx_ = other.base_idx_;
    qual_idx_ = other.qual_idx_;
    meta_idx_ = other.meta_idx_;

    base_data_ = other.base_data_;
    qual_data_ = other.qual_data_;
    meta_data_ = other.meta_data_;

    num_records_ = other.num_records_;
    record_idx_ = other.record_idx_;

    other.bases_ = nullptr;
    other.quals_ = nullptr;
    other.meta_ = nullptr;

    other.base_data_ = nullptr;
    other.qual_data_ = nullptr;
    other.meta_data_ = nullptr;

    other.base_idx_ = nullptr;
    other.qual_idx_ = nullptr;
    other.meta_idx_ = nullptr;
    return *this;
  }

  bool AGDReadResource::reset_iter()
  {
    record_idx_ = 0;
    auto idx_offset = num_records_ * sizeof(RecordTable::IndexValue);
    base_data_ = bases_->get()->data() + idx_offset;
    qual_data_ = quals_->get()->data() + idx_offset;
    if (has_metadata()) {
      meta_data_ = meta_->get()->data() + idx_offset;
    }

    return true;
  }


  bool AGDReadResource::has_qualities()
  {
    return quals_ != nullptr;
  }

  bool AGDReadResource::has_metadata()
  {
    return meta_ != nullptr;
  }

  Status AGDReadResource::get_next_record(const char **bases, std::size_t *bases_length,
                                            const char **qualities, std::size_t *qualities_length)
  {
    if (record_idx_ < num_records_) {
      auto base_len = base_idx_->relative_index[record_idx_];
      *bases_length = base_len;
      *bases = base_data_;
      base_data_  += base_len;

      auto qual_len = qual_idx_->relative_index[record_idx_];
      *qualities_length = qual_len;
      *qualities = qual_data_;
      qual_data_ += qual_len;

      record_idx_++;
      return Status::OK();
    } else {
      return ResourceExhausted("agd record container exhausted");
    }

  }

  Status AGDReadResource::get_next_record(const char **bases, std::size_t *bases_length,
                                            const char **qualities, std::size_t *qualities_length,
                                            const char **metadata, std::size_t *metadata_length)
  {
    if (record_idx_ < num_records_) {
      auto base_len = base_idx_->relative_index[record_idx_];
      *bases_length = base_len;
      *bases = base_data_;
      base_data_  += base_len;

      auto qual_len = qual_idx_->relative_index[record_idx_];
      *qualities_length = qual_len;
      *qualities = qual_data_;
      qual_data_ += qual_len;

      auto meta_len = meta_idx_->relative_index[record_idx_];
      *metadata_length = meta_len;
      *metadata = meta_data_;
      meta_data_ += meta_len;

      record_idx_++;
      return Status::OK();
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
  }

  void AGDReadResource::release() {
    if (bases_) {
      bases_->release();
    }
    if (quals_) {
      quals_->release();
    }
    if (meta_) {
      meta_->release();
    }
  }

  Status AGDReadResource::split(size_t chunk, vector<unique_ptr<ReadResource>> &split_resources) {
    split_resources.clear();

    reset_iter(); // who cares doesn't die for now

    // TODO we don't support dealing with meta for now. too difficult for the deadline!
    decltype(base_data_) base_start = base_data_, qual_start = qual_data_, meta_start = nullptr;
    
    decltype(chunk) max_range;
    for (decltype(num_records_) i = 0; i < num_records_; i += chunk) {
      //AGDReadSubResource a(*this, i, CHUNK_SIZE, )
      max_range = i + chunk;
      if (max_range > num_records_) {
        // deals with the tail
        max_range = num_records_;
      }

      unique_ptr<ReadResource> a(new AGDReadSubResource(*this, i, max_range, base_start, qual_start, meta_start));
      split_resources.push_back(move(a));

      // actually advance the records
      for (decltype(i) j = i; j < max_range; ++j) {
        base_start += base_idx_->relative_index[j];
        qual_start += qual_idx_->relative_index[j];
      }
    }
    return Status::OK();
  }

  AGDReadSubResource::AGDReadSubResource(const AGDReadResource &parent_resource,
                                             size_t index_offset, size_t max_idx,
                                             const char *base_data_offset, const char *qual_data_offset, const char *meta_data_offset) : start_idx_(index_offset), max_idx_(max_idx), current_idx_(index_offset),
                                                                                                                                         base_idx_(parent_resource.base_idx_),
                                                                                                                                         qual_idx_(parent_resource.qual_idx_),
                                                                                                                                         meta_idx_(parent_resource.meta_idx_),
                                                                                                                                         base_data_(base_data_offset), base_start_(base_data_offset),
                                                                                                                                         qual_data_(qual_data_offset), qual_start_(qual_data_offset),
                                                                                                                                         meta_data_(meta_data_offset), meta_start_(meta_data_offset) {}

  Status AGDReadSubResource::get_next_record(const char **bases, size_t *bases_length,
                                               const char **qualities, size_t *qualities_length,
                                               const char **metadata, size_t *metadata_length)
  {
    return Unimplemented("AGDReadSubResource doesn't implement get_next_record with metadata");
  }

  Status AGDReadSubResource::get_next_record(const char **bases, size_t *bases_length,
                                               const char **qualities, size_t *qualities_length)
  {
    if (current_idx_ < max_idx_) {
      auto base_len = base_idx_->relative_index[current_idx_];
      *bases_length = base_len;
      *bases = base_data_;
      base_data_  += base_len;

      auto qual_len = qual_idx_->relative_index[current_idx_++];
      *qualities_length = qual_len;
      *qualities = qual_data_;
      qual_data_ += qual_len;

      return Status::OK();
    } else {
      return ResourceExhausted("agd record container exhausted");
    }
  }

  bool AGDReadSubResource::reset_iter() {
    base_data_ = base_start_;
    qual_data_ = qual_start_;
    meta_data_ = meta_start_;
    current_idx_ = start_idx_;
    return true;
  }

  bool AGDReadSubResource::has_qualities()
  {
    return qual_start_ != nullptr;
  }

  bool AGDReadSubResource::has_metadata()
  {
    return meta_start_ != nullptr;
  }

  size_t AGDReadResource::num_records() {
    return num_records_;
  }

  size_t AGDReadSubResource::num_records() {
    return max_idx_ - start_idx_;
  }

} // namespace tensorflow {
