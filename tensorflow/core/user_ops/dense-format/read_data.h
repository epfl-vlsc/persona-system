#ifndef TENSORFLOW_CORE_USEROPS_DNA_ALIGN_DATA_H_
#define TENSORFLOW_CORE_USEROPS_DNA_ALIGN_DATA_H_

#include <cstdint>
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

  class ReadData : public ResourceBase {
  public:
    virtual ~ReadData();

    virtual bool has_metadata();
    virtual std::size_t num_records() = 0;

    virtual Status qualities(std::size_t index, const char **data, std::size_t *length) = 0;
    virtual Status bases(std::size_t index, const char **data, std::size_t *length) = 0;
    virtual Status metadata(std::size_t index, const char **data, std::size_t *length);

    virtual Status get_next_record(const char **bases, std::size_t *bases_length,
                                   const char **qualities, std::size_t *qualities_length);
    virtual Status get_next_record(const char **bases, std::size_t *bases_length,
                                   const char **qualities, std::size_t *qualities_length,
                                   const char **metadata, std::size_t *metadata_length);

    virtual void reset_iter();

    virtual string DebugString() override;
  protected:
    std::size_t iter_ = 0;
    bool exhausted();

  private:
    Status get_current_record(const char **bases, std::size_t *bases_length,
                              const char **qualities, std::size_t *qualities_length);
  };
} // namespace tensorflow {

#endif
