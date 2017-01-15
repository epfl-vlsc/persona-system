#pragma once

#include "agd_record_reader.h"


namespace tensorflow {

  using namespace errors;

  /*
   * A class that provides a "view" over the result data in the resource container.
   * Does not take ownership of the underlying data
   *
   * NB: The result reader expects the dataset to be sorted by GenomeLocation.
   * GetRecordAtLocation() will not work correctly otherwise.
   */
  class AGDResultReader : public AGDRecordReader {
  public:
    // metadata column is required to disambiguate results that mapped to 
    // the same position.
    AGDResultReader(const char* resource, size_t num_records, AGDRecordReader* metadata);

    // Get the result at specified GenomeLocation. Uses a binary search
    // for log(n) performance. metadata may be used to disambiguate reads
    // that have mapped to the same position, which is likely. 
    // NB cigar is not a proper c-string
    Status GetResultAtLocation(int64_t location, const char* metadata, 
        size_t metadata_len, const format::AlignmentResult** result, const char** cigar, size_t* cigar_len);

    // Get or peek next alignment result in the index. 
    // NB again cigar is not a proper c-string
    Status GetNextResult(const format::AlignmentResult** result, const char** cigar, 
        size_t* cigar_len);
    Status PeekNextResult(const format::AlignmentResult** result, const char** cigar, 
        size_t* cigar_len);

    // Get a result at a specific index offset in the chunk
    // uses the Absolute index created on construction
    Status GetResultAtIndex(size_t index, const format::AlignmentResult** result,
        const char** cigar, size_t* cigar_len);

    // is this location possibly contained
    // i.e. start_location_ <= location <= end_location_
    bool IsPossiblyContained(int64_t location) {
      return location >= start_location_ && location <= end_location_;
    }

  private:
    int64_t start_location_;
    int64_t end_location_;
    AGDRecordReader* metadata_;

  };

} // namespace TF
