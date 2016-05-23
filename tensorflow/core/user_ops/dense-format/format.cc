#include "format.h"

namespace tensorflow {
namespace format {

using namespace std;

Status BinaryBaseRecord::appendToVector(const std::size_t record_size_in_bytes, vector<char> &output, vector<char> &lengths) const
{
  if (record_size_in_bytes % sizeof(uint64_t) != 0) {
    return errors::InvalidArgument("Size of record ", record_size_in_bytes, " is not a multiple of ", sizeof(uint64_t));
  }

  Status status;
  const size_t record_size_in_base_records = record_size_in_bytes / 8;
  char base_char = '\0';
  uint8_t length; // TODO may need to deal with size_t issue better!
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

      length++;
      output.push_back(base_char);
    }
  }
  lengths.push_back(static_cast<char>(length));
  return Status::OK();
}

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

Status
BinaryBaseRecord::IntoBases(const char *fastq_base, const std::size_t fastq_base_size, std::vector<BinaryBases> &bases)
{
  size_t base_idx = 0;
  BinaryBases bb;
  for (size_t i = 0; i < fastq_base_size; i++) {
    TF_RETURN_IF_ERROR(bb.setBase(fastq_base[i], base_idx++));
    if (base_idx == bb.compression) {
      bases.push_back(bb); // this should be a copy, by C++ default behavior
      bb.bases = 0;
      base_idx = 0;
    }
  }

  bb.terminate(base_idx);
  bases.push_back(bb);
}

Status BinaryBases::setBase(const char base, size_t position) {
  BaseAlphabet b;
  switch (base) {
  case 'a':
  case 'A':
    b = BaseAlphabet::A;
    break;
  case 'c':
  case 'C':
    b = BaseAlphabet::C;
    break;
  case 'g':
  case 'G':
    b = BaseAlphabet::G;
    break;
  case 't':
  case 'T':
    b = BaseAlphabet::T;
    break;
  case 'n':
  case 'N':
    b = BaseAlphabet::N;
    break;
  default:
    return errors::InvalidArgument("Unable to convert the following base character: ", string(&base, 1));
  }

  return setBaseAtPosition(b, position);
}

Status BinaryBases::terminate(const size_t position) {
  return setBaseAtPosition(BaseAlphabet::END, position);
}

Status BinaryBases::setBaseAtPosition(const BaseAlphabet base, const size_t position) {
  assert(position < compression);
  if (position < compression) {
    uint64_t packed_base = 0x7ull & static_cast<uint64_t>(base);// get the lower bits only!
    size_t packed_shift = base_width * position;
    uint64_t clear_mask = 0x7ull << packed_shift;
    bases &= ~clear_mask;
    bases |= packed_base << packed_shift;
    return Status::OK();
  } else {
    return errors::InvalidArgument("Unable to set base position of ", position, "\nMaximum position: ", compression-1);
  }
}

} // namespace format
} // namespace tensorflow
