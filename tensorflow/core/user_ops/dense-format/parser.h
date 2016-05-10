#ifndef TENSORFLOW_CORE_USER_OPS_DENSE_FORMAT_PARSER_H_
#define TENSORFLOW_CORE_USER_OPS_DENSE_FORMAT_PARSER_H_

#include "format.h"
#include "tensorflow/core/lib/core/errors.h"
#include <vector>
#include <string>
#include <cstdint>

namespace tensorflow {
  class RecordParser
  {
  public:
    Status ParseNew(const char* data, const std::size_t length);

    size_t RecordCount();

    bool HasNextRecord();

    Status GetNextRecord(std::string *value);

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
