#ifndef TENSORFLOW_CORE_USEROPS_DENSE_DATA_H_
#define TENSORFLOW_CORE_USEROPS_DENSE_DATA_H_

#include <memory>
#include "parser.h"
#include "tensorflow/core/user_ops/dna-align/read_data.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

  class DenseReadData : public ReadData {
  public:
    typedef RecordParser* RecordBuffer;
    DenseReadData(RecordBuffer bases,
                  RecordBuffer qualities,
                  RecordBuffer metadata = nullptr);

    virtual ~DenseReadData();

    // TODO override destructor here for proper cleanup
    virtual Status qualities(std::size_t index, const char **data, std::size_t *length) override;
    virtual Status bases(std::size_t index, const char **data, std::size_t *length) override;
    virtual Status metadata(std::size_t index, const char **data, std::size_t *length) override;

    virtual bool has_metadata() override;
    virtual std::size_t num_records() override;

    // TODO have a node that specifically sets the metadata on an existing upstream node
    Status set_metadata(RecordBuffer metadata);

    virtual Status get_next_record(const char **bases, std::size_t *bases_length,
                                   const char **qualities, std::size_t *qualities_length) override;
    virtual Status get_next_record(const char **bases, std::size_t *bases_length,
                                   const char **qualities, std::size_t *qualities_length,
                                   const char **metadata, std::size_t *metadata_length) override;

  private:
    RecordBuffer bases_, qualities_, metadata_;
  };
} // namespace tensorflow {

#endif
