#include "format.h"
#include "parser.h"
#include "util.h"
#include <array>
#include <cstring>
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace format {

using namespace std;
using namespace errors;

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

Status append(const BinaryBases* bases, const std::size_t record_size_in_bytes, Buffer &data, Buffer &lengths)
{
  if (record_size_in_bytes % sizeof(uint64_t) != 0) {
    return InvalidArgument("Size of record ", record_size_in_bytes, " is not a multiple of ", sizeof(uint64_t));
  }

  Status status;
  const size_t record_size_in_base_records = record_size_in_bytes / 8;
  size_t length = 0;
  for (size_t i = 0; i < record_size_in_base_records; ++i) {
    auto const base = &bases[i];
    TF_RETURN_IF_ERROR(base->append(data, &length));
  }
  RelativeIndex char_len = static_cast<RelativeIndex>(length);
  lengths.AppendBuffer(reinterpret_cast<const char*>(&char_len), sizeof(char_len));
  return Status::OK();
}

Status BinaryBases::append(Buffer &data, size_t *num_bases) const
{
  using namespace errors;
  auto bases_copy = bases;
  size_t length = 0;
  //TODO this is the method to fix
  for (size_t i = 0; i < compression;) {
    auto res = lookup_triple(bases_copy);
    if (res == nullptr) {
      return Internal("unable to convert value ", bases_copy & 0x1ff, " to a triple\n");
    }

    auto val = res->get();
    auto num_chars = res->effective_characters();
    i += val.size();

    data.AppendBuffer(val.data(), num_chars);
    length += num_chars;
    // there must be a terminating character in here
    auto sz = val.size();
    if (num_chars < sz) {
      break;
    } else {
      bases_copy >>= sz * base_width;
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
    return ResourceExhausted("done");
  default:
    return NotFound("Base alphabet for type ", x, " not found");
  }
  return Status::OK();
}

Status
IntoBases(const char *fastq_base, const std::size_t fastq_base_size, std::vector<BinaryBases> &bases)
{
  bases.clear();
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
  return Status::OK();
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
    return InvalidArgument("Unable to convert the following base character: ", string(&base, 1),
    " are you sure this column is base pair data?");
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
    return InvalidArgument("Unable to set base position of ", position, "\nMaximum position: ", compression-1);
  }
}

} // namespace format
} // namespace tensorflow
