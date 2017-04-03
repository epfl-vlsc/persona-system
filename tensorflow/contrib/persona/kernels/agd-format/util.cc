#include "util.h"
#include <cstring>
#include <cstdint>
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
  using namespace std;

  void DataResourceReleaser(ResourceContainer<Data> *data) {
    core::ScopedUnref a(data);
    {
      ResourceReleaser<Data> a1(*data);
      {
        DataReleaser dr(*data->get());
      }
    }
  }
} // namespace tensorflow {
