#include "data.h"

namespace tensorflow {
  DataReleaser::DataReleaser(Data &data) : data_(data) {}
  DataReleaser::~DataReleaser() {
    data_.release();
  }
} // namespace tensorflow {
