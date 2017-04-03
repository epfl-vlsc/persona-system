#pragma once

#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "data.h"
#include "tensorflow/contrib/persona/kernels/object-pool/resource_container.h"

namespace tensorflow {

  template <typename T>
    inline void safe_reserve(std::vector<T> &v, const std::size_t ideal_length, const size_t additive_increase_bonus_size = 2 * 1024 * 1024) {
    if (v.capacity() < ideal_length) {
      v.reserve(ideal_length + additive_increase_bonus_size);
    }
  }

  void DataResourceReleaser(ResourceContainer<Data> *data);
} // namespace tensorflow {
