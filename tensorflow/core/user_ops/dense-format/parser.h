#ifndef TENSORFLOW_CORE_USER_OPS_DENSE_FORMAT_PARSER_H_
#define TENSORFLOW_CORE_USER_OPS_DENSE_FORMAT_PARSER_H_

#include "format.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include <vector>
#include <array>
#include <string>
#include <cstdint>

namespace tensorflow {

  template <size_t N>
  class BaseMapping {

  private:
    std::array<char, N> characters_;
    std::size_t effective_characters_;

  public:

    BaseMapping(std::array<char, N> chars, std::size_t effective_characters) : characters_(chars),
      effective_characters_(effective_characters) {}

    BaseMapping() {
      characters_.fill('\0');
      characters_[0] = 'Z'; // TODO hack: an arbitrary bad value, used to indicate an impossible issue
    }

    const std::array<char, N>& get() const {
      return characters_;
    }

    const std::size_t effective_characters() const {
      return effective_characters_;
    }
  };

  const BaseMapping<3>*
  lookup_triple(const std::size_t bases);

  class RecordParser
  {
  public:

    Status ParseNew(const char* data, const std::size_t length, const bool verify, std::vector<char> &result_buffer);

  private:

    void reset();

    std::vector<char> conversion_scratch_, index_scratch_;
    const format::RecordTable *records = nullptr;
  };

}  //  namespace tensorflow {

#endif
