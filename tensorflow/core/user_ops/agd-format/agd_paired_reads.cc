#include "agd_paired_reads.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
  using namespace std;
  using namespace errors;

  AGDPairedReadResource::AGDPairedReadResource(std::size_t num_records, DataContainer *bases, DataContainer *quals) {
    if (num_records % 2 != 0) {
      LOG(ERROR) << "Attempting to construct a read resource with an odd number of reads!";
      throw exception();
    }
    num_records_ = num_records;
    bases_ = bases;
    quals_ = quals;
    auto idx_offset = num_records * sizeof(format::RelativeIndex);
    auto b = bases->get()->data();
    auto q = quals->get()->data();
    base_idx_ = reinterpret_cast<const format::RelativeIndex*>(b);
    qual_idx_ = reinterpret_cast<const format::RelativeIndex*>(q);
    base_data_ = b + idx_offset;
    qual_data_ = q + idx_offset;
  }

  AGDPairedReadResource&
  AGDPairedReadResource::operator=(AGDPairedReadResource && other)
  {
    bases_ = other.bases_;
    quals_ = other.quals_;
    base_idx_ = other.base_idx_;
    qual_idx_ = other.qual_idx_;
    num_records_ = other.num_records_;
    record_idx_ = other.record_idx_;
    base_data_ = other.base_data_;
    qual_data_ = other.qual_data_;
    sub_resources_ = move(other.sub_resources_);
    sub_resource_index_ = other.sub_resource_index_.exchange(0, memory_order_relaxed);

    other.bases_ = nullptr;
    other.quals_ = nullptr;
    other.base_idx_ = nullptr;
    other.qual_idx_ = nullptr;
    other.base_data_ = nullptr;
    other.qual_data_ = nullptr;
    return *this;
  }

  void AGDPairedReadResource::release() {
    if (bases_) {
      bases_->release();
    }
    if (quals_) {
      quals_->release();
    }
    sub_resources_.clear();
    buffer_list_ = nullptr;
  }

  Status AGDPairedReadResource::get_next_subchunk(PairedReadResource **rr, BufferPair **b) {
    auto a = sub_resource_index_.fetch_add(1, memory_order_relaxed);
    if (a >= sub_resources_.size()) {
      return ResourceExhausted("No more AGD pair subchunks");
    } else if (a == 0) {
      buffer_list_->set_start_time();
    }

    *rr = &sub_resources_[a];
    *b = &(*buffer_list_)[a];
    return Status::OK();
  }

  // TODO should it be chunk or 2x chunk?
  Status AGDPairedReadResource::split(size_t chunk, BufferList *bl) {
    sub_resources_.clear();
    reset_iter();
    decltype(base_data_) base_start = base_data_, qual_start = qual_data_;
    chunk *= 2;
    decltype(chunk) max_range;

    for (decltype(num_records_) i = 0; i < num_records_; i += chunk) {
      max_range = i + chunk;
      if (max_range > num_records_) {
        max_range = num_records_;
      }

      // TODO fill in params
      sub_resources_.push_back(AGDPairedReadSubResource(*this, i, max_range, base_start, qual_start));

      for (auto j = i; j < max_range; ++j) {
        base_start += base_idx_[j];
        qual_start += qual_idx_[j];
      }
    }

    sub_resource_index_.store(0, memory_order_relaxed);
    bl->resize(sub_resources_.size());
    buffer_list_ = bl;
    return Status::OK();
  }

  size_t AGDPairedReadResource::num_records() {
    return num_records_;
  }

  Status AGDPairedReadResource::get_next_record(std::array<std::pair<const char*, std::size_t>, 2> &bases, std::array<std::pair<const char*, std::size_t>, 2> &qualities) {
    if (record_idx_ < num_records_ - 1) {
      for (size_t i = 0; i < 2; ++i, ++record_idx_) {
        auto base_len = base_idx_[record_idx_];
        auto qual_len = qual_idx_[record_idx_];
        bases[i] = make_pair(base_data_, base_len);
        qualities[i] = make_pair(qual_data_, qual_len);
        base_data_ += base_len;
        qual_data_ += qual_len;
      }
    } else {
      return ResourceExhausted("agd paired record container exhausted");
    }
    return Status::OK();
  }

  bool AGDPairedReadResource::reset_iter() {
    record_idx_ = 0;
    auto idx_offset = num_records_ * sizeof(format::RelativeIndex);
    base_data_ = bases_->get()->data() + idx_offset;
    qual_data_ = quals_->get()->data() + idx_offset;
    return true;
  }

  AGDPairedReadSubResource::AGDPairedReadSubResource(const AGDPairedReadResource &parent_resource,
                                                     size_t index_offset, size_t max_idx,
                                                     const char *base_data_offset, const char *qual_data_offset) : max_idx_(max_idx), current_idx_(index_offset), start_idx_(index_offset),
                                                                                                                   base_idx_(parent_resource.base_idx_), qual_idx_(parent_resource.qual_idx_),
                                                                                                                   base_data_(base_data_offset), base_start_(base_data_offset),
                                                                                                                   qual_data_(qual_data_offset), qual_start_(qual_data_offset) {}

  Status AGDPairedReadSubResource::get_next_record(array<pair<const char*, size_t>, 2> &bases, array<pair<const char*, size_t>, 2> &qualities) {
    if (current_idx_ < max_idx_ - 1) {
      for (size_t i = 0; i < 2; ++i, ++current_idx_) {
        auto base_len = base_idx_[current_idx_];
        auto qual_len = qual_idx_[current_idx_];
        bases[i] = make_pair(base_data_, base_len);
        qualities[i] = make_pair(qual_data_, qual_len);
        base_data_ += base_len;
        qual_data_ += qual_len;
      }
      return Status::OK();
    } else {
      return ResourceExhausted("agd sub read resource exhausted");
    }
  }

  size_t AGDPairedReadSubResource::num_records() {
    return (max_idx_ - start_idx_) / 2;
  }
} // namespace tensorflow {
