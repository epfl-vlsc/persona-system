#ifndef TENSORFLOW_CORE_USEROPS_DENSE_DATA_H_
#define TENSORFLOW_CORE_USEROPS_DENSE_DATA_H_

#include <memory>
#include "parser.h"
#include "tensorflow/core/user_ops/dna-align/data.h"

namespace tensorflow {

  class DenseReadData : public ReadData {
  public:
    typedef std::shared_ptr<RecordParser> RecordBuffer;
    DenseReadData(RecordBuffer bases,
                  RecordBuffer qualities,
                  RecordBuffer metadata = nullptr);

    // TODO override destructor here for proper cleanup

    virtual bool has_metadata() override;
    virtual std::size_t num_records() override;

    virtual const char* qualities(std::size_t index) override;
    virtual const char* bases(std::size_t index) override;
    virtual std::size_t bases_length(std::size_t index) override;

    virtual const char* metadata(std::size_t index) override;
    virtual std::size_t metadata_length(std::size_t index) override;

    void set_metadata(RecordBuffer metadata);

  private:
    RecordBuffer bases_, qualities_, metadata_;
  };
} // namespace tensorflow {

#endif
