#ifndef TENSORFLOW_CORE_USEROPS_DNA_ALIGN_DATA_H_
#define TENSORFLOW_CORE_USEROPS_DNA_ALIGN_DATA_H_

#include <cstdint>
#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {

  class ReadData : public ResourceBase {
  public:
    virtual ~ReadData();

    virtual bool has_metadata();
    virtual std::size_t num_records() = 0;

    // The expensive methods, to access by index
    virtual const char* qualities(std::size_t index) = 0;
    virtual const char* bases(std::size_t index) = 0;
    virtual std::size_t bases_length(std::size_t index) = 0;
    // Note: we assume bases and qualities are the same length (they should be)

    virtual const char* metadata(std::size_t index);
    virtual std::size_t metadata_length(std::size_t index);

    virtual bool get_next_record(const char **bases, std::size_t *bases_length,
                                 const char **qualities);

    virtual bool get_next_record(const char **bases, std::size_t *bases_length,
                                 const char **qualities,
                                 const char **metadata, std::size_t *metadata_length);

    virtual void reset_iter();
  protected:
    std::size_t iter_ = 0;

  private:
    bool get_current_record(const char **bases, std::size_t *bases_length,
                                  const char **qualities);
  };
} // namespace tensorflow {

#endif
