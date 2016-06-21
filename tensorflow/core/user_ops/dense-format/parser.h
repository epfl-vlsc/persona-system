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

  public:

    BaseMapping(std::array<char, N> chars) : characters_(chars) {}

    BaseMapping() {
      characters_.fill('\0');
      characters_[0] = 'Z'; // TODO hack: an arbitrary bad value, used to indicate an impossible issue
    }

    const std::array<char, N>& get() const {
      return characters_;
    }
  };

  const BaseMapping<3>*
  lookup_triple(const std::size_t bases);

  class RecordParser
  {
  public:
    RecordParser(std::size_t size);
    RecordParser() = default;

    Status ParseNew(const char* data, const std::size_t length, const bool verify, std::vector<char> &scratch, std::vector<char> &index_scratch);

    size_t RecordCount();

    bool HasNextRecord();

    Status GetNextRecord(const char** value, std::size_t *length);

    Status GetRecordAtIndex(std::size_t index, const char **value, std::size_t *length);

    void ResetIterator();

  private:

    void reset();

    std::vector<char> buffer_;
    format::FileHeader file_header_;
    const format::RecordTable *records = nullptr;
    bool valid_record_ = false;
    size_t total_records_ = 0;
    size_t current_record_ = 0;
    size_t current_offset_ = 0;
  };

}  //  namespace tensorflow {

#endif
