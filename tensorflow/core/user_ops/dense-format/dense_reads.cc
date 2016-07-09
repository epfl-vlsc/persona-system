#include "dense_reads.h"

namespace tensorflow {
  DenseReadResource::DenseReadResource(DataContainer *bases, DataContainer *quals, DataContainer *meta) :
    bases_(bases), quals_(quals), meta_(meta)
  {}
  
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

  Status get_next_record(const char **bases, std::size_t *bases_length,
                         const char **qualities, std::size_t *qualities_length,
                         const char **metadata, std::size_t *metadata_length)
  {
    return Status::OK();
  }

} // namespace tensorflow {
