#include "dense_data.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
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

  void DenseReadData::set_metadata(RecordBuffer metadata) {
    metadata_ = metadata;
  }

  bool DenseReadData::has_metadata() {
    return metadata_.get() != nullptr;
  }

  std::size_t DenseReadData::num_records() {
    return bases_->RecordCount();
  }


  const char* DenseReadData::qualities(std::size_t index) {
    
  }

  const char* DenseReadData::bases(std::size_t index) {

  }

  std::size_t DenseReadData::bases_length(std::size_t index) {

  }


  const char* DenseReadData::metadata(std::size_t index) {

  }

  std::size_t DenseReadData::metadata_length(std::size_t index) {

  }

} // namespace tensorflow {
