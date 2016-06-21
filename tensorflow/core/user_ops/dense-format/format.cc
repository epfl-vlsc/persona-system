#include "format.h"
#include <array>
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace format {

using namespace std;

namespace {
  struct BaseMap {
    const BaseAlphabet base;
    const char base_char;
    constexpr BaseMap(BaseAlphabet base_, char base_char_) : base(base_), base_char(base_char_) {}
  };

  const std::array<BaseMap, 5> base_map {{
    BaseMap(BaseAlphabet::A, 'A'),
    BaseMap(BaseAlphabet::G, 'G'),
    BaseMap(BaseAlphabet::C, 'C'),
    BaseMap(BaseAlphabet::T, 'T'),
    BaseMap(BaseAlphabet::N, 'N'),
  }};
}

Status BinaryBaseRecord::appendToVector(const std::size_t record_size_in_bytes, vector<char> &output, vector<char> &lengths) const
{
  if (record_size_in_bytes % sizeof(uint64_t) != 0) {
    return errors::InvalidArgument("Size of record ", record_size_in_bytes, " is not a multiple of ", sizeof(uint64_t));
  }

  Status status;
  const size_t record_size_in_base_records = record_size_in_bytes / 8;
  size_t length = 0;
  for (size_t i = 0; i < record_size_in_base_records; ++i) {
    auto const base = &bases[i];
    TF_RETURN_IF_ERROR(base->appendToVector(output, &length));
  }
  lengths.push_back(static_cast<char>(length));
  return Status::OK();
}

Status BinaryBases::appendToVector(vector<char> &output, size_t *num_bases) const
{
  using namespace errors;
  BaseAlphabet base;
  auto bases_copy = bases;
  uint64_t base_i;
  size_t length = 0;
  bool set;
  //TODO this is the method to fix
  for (size_t i = 0; i < compression; i+=3) {
    base_i = (bases_copy & 0x1ff);
    // TODO need to do something with length here

    set = false;
    for (const auto &bm : base_map) {
      if (base == bm.base) {
        output.push_back(bm.base_char);
        set = true;
        break;
      }
    }

    if (!set) {
      // Don't worry about unwinding this now
      return Internal("Could not find conversion for base ", static_cast<int>(base));
    } else {
      bases_copy >>= 3;
    }
  }

  *num_bases += length;

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
