#include "format.h"

namespace tensorflow {
namespace format {

using namespace std;

Status
BinaryBaseRecord::toString(const size_t record_size_in_bytes, string *output) const
{
  if (record_size_in_bytes % sizeof(uint64_t) != 0) {
    return errors::InvalidArgument("Size of record ", record_size_in_bytes, " is not a multiple of ", sizeof(uint64_t));
  }
  const size_t record_size_in_base_records = record_size_in_bytes / 8;
  string converted;
  Status status;
  char base_char = '\0';
  for (size_t i = 0; i < record_size_in_base_records; ++i) {
    auto const base = &bases[i];
    for (size_t j = 0; j < BinaryBases::compression; ++j) {
      status = base->getBase(j, &base_char);

      if (!status.ok()) {
        if (errors::IsResourceExhausted(status)) {
          break; // need to break because of last one
        } else {
          return status;
        }
      }

      converted.push_back(base_char);
    }
  }

  *output = converted;
  return Status::OK();
}

Status
BinaryBases::getBase(const size_t position, char* base) const
{
  uint64_t x = bases >> position * base_width;
  x &= 0x7ull;
  auto b = static_cast<BaseAlphabet>(x);
  switch(b) {
  case A:
    *base = 'A';
    break;
  case C:
    *base = 'C';
    break;
  case G:
    *base = 'G';
    break;
  case T:
    *base = 'T';
    break;
  case N:
    *base = 'N';
    break;
  case END:
    return errors::ResourceExhausted("done");
  default:
    return errors::NotFound("Base alphabet for type ", x, " not found");
  }
  return Status::OK();
}

} // namespace format
} // namespace tensorflow
