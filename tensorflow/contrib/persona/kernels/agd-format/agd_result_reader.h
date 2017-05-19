#pragma once

#include "agd_record_reader.h"
#include <utility>
#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"

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

    // class Position is defined by alignment.pb.h

    // metadata column is required to disambiguate results that mapped to 
    // the same position.
    // if null is passed, GetResultAtLocation will not work.
    AGDResultReader(const char* resource, size_t num_records, AGDRecordReader* metadata=nullptr);
    AGDResultReader(ResourceContainer<Data>* resource, size_t num_records, AGDRecordReader* metadata=nullptr);

    // Get the result at specified GenomeLocation. Uses a binary search
    // for log(n) performance. metadata may be used to disambiguate reads
    // that have mapped to the same position, which is likely. Also returns
    // the index position the read was found at.
    Status GetResultAtPosition(Position& position, const char* metadata,
        size_t metadata_len, Alignment& result, size_t* index=nullptr);

    // Get or peek next alignment result in the index. 
    // NB again cigar is not a proper c-string
    Status GetNextResult(Alignment& result);
    Status PeekNextResult(Alignment& result);

    // Get a result at a specific index offset in the chunk
    // uses the Absolute index created on construction
    Status GetResultAtIndex(size_t index, Alignment& result);

    // is this location possibly contained
    // i.e. start_location_ <= location <= end_location_
    bool IsPossiblyContained(Position& position) {
      return position.ref_index() >= start_position_.ref_index() && position.ref_index() <= end_position_.ref_index()
              && position.position() >= start_position_.position() && position.position() <= end_position_.position();
    }

  private:

    Position start_position_;
    Position end_position_;
    AGDRecordReader* metadata_;

  };

} // namespace TF
