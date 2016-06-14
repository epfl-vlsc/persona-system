#ifndef TENSORFLOW_CORE_USER_OPS_DENSE_FORMAT_PARSER_H_
#define TENSORFLOW_CORE_USER_OPS_DENSE_FORMAT_PARSER_H_

#include "format.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include <vector>
#include <string>
#include <cstdint>

namespace tensorflow {
  class RecordParser
  {
  public:
    RecordParser(std::size_t size);
    RecordParser() = default;

    Status ParseNew(const char* data, const std::size_t length, const bool verify);

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
