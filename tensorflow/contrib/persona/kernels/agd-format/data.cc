#include "data.h"

namespace tensorflow {
  DataReleaser::DataReleaser(Data &data) : data_(data) {}
  DataReleaser::~DataReleaser() {
    data_.release();
  }

  void Data::release() {}

  char* Data::mutable_data() {
    return nullptr;
  }
} // namespace tensorflow {
