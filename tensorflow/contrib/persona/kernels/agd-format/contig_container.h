#pragma once

#include <vector>
#include <stdexcept>

/*
 * This is just a wrapper around vector, but avoids any reallocation due to resize() or
 * other such calls to vector<T>
 *
 * Only supports "appending" and then subsequent access
 */

template <typename T>
class ContigContainer {
private:
  std::size_t size_ = 0;
  std::vector<T> container_;
public:
  void clear() {
    size_ = 0;
  }

  void push_back(const T &t) {
    if (size_ == container_.size()) {
      container_.push_back(t);
    } else {
      container_[size_] = t;
    }
    ++size_;
  }

  void resize(const std::size_t new_size) {
    if (new_size > container_.size()) {
      container_.resize(new_size);
    }
    size_ = new_size;
  }

  decltype(size_) size() {
    return size_;
  }

  T& operator[](std::size_t index) {
    if (index <= size_) {
      return container_[index];
    }
    throw std::out_of_range("Unable to index into contiguous container");
  }
};