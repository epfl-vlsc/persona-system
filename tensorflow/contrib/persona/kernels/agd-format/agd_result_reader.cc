
#include "agd_result_reader.h"

namespace tensorflow {

  using namespace errors;
  using namespace std;
  using namespace format;

  AGDResultReader::AGDResultReader(const char* resource, size_t num_records, 
      AGDRecordReader* metadata) : AGDRecordReader(resource, num_records), metadata_(metadata) {
    Alignment result;
    Status s = PeekNextResult(result);
    start_position_ = result.position();
    GetResultAtIndex(num_records - 1, result);
    end_position_ = result.position();
    if (metadata_)
      metadata_->Reset();
    /*LOG(INFO) << "AGDResult reader has chunk with first: " << start_position_.DebugString() << " and last: "
              << end_position_.DebugString();*/
  }
  
  AGDResultReader::AGDResultReader(ResourceContainer<Data>* resource, size_t num_records, 
      AGDRecordReader* metadata) : AGDRecordReader(resource, num_records), metadata_(metadata) {
    Alignment result;
    Status s = PeekNextResult(result);
    start_position_ = result.position();
    GetResultAtIndex(num_records - 1, result);
    end_position_ = result.position();
    if (metadata_)
      metadata_->Reset();

    /*LOG(INFO) << "AGDResult reader has chunk with first: " << start_position_.DebugString() << " and last: "
              << end_position_.DebugString();*/
  }

  inline bool operator<(const Position& lhs, const Position& rhs) {
    if (lhs.ref_index() < rhs.ref_index()) {
      return true;
    } else if (lhs.ref_index() == rhs.ref_index()) {
      if (lhs.position() < rhs.position()) return true;
      else return false;
    } else
      return false;
  }

  inline bool operator==(const Position& lhs, const Position& rhs) {
    return (lhs.ref_index() == rhs.ref_index() && lhs.position() == rhs.position());
  }

  Status AGDResultReader::GetResultAtPosition(Position& position, const char* metadata,
        size_t metadata_len, Alignment& result, size_t* index) {

    // this method should only be used on the primary results column
    // empty results will cause it to return with errors::Unavailable
    if (!metadata_)
      return Internal("metadata was not supplied so GetResultAtLocation cannot work.");

    if (!IsPossiblyContained(position))  // caller may have to get next chunk to find
      return NotFound("Location ", position.DebugString(), " is out of the genome location bounds ",
          "of this chunk. Are you sure the set is sorted?");

    // the idea is to find the first occurrence of result with `location` using bSearch, 
    // and then loop through all results with `location` to find the one that matches `metadata`
    // the metadata of paired reads or read segments should match

    size_t high = num_records_ - 1;
    size_t low = 0;
    size_t first = 0xffffffff; // guessing a chunk will not have 4 billion records
    while (low < high) {
      size_t mid = low + ((high-low) / 2);

      TF_RETURN_IF_ERROR(GetResultAtIndex(mid, result));
      if (result.position() == position) {
        // a match, but keep going to find the first occurrence
        first = mid;
        high = mid - 1;
      } else if (result.position() < position) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    if (first == 0xffffffff)
      return NotFound("Location ", position.DebugString(), " was not found.");

    const char* meta_str;
    size_t meta_len;
    TF_RETURN_IF_ERROR(metadata_->GetRecordAt(first, &meta_str, &meta_len));
    TF_RETURN_IF_ERROR(GetResultAtIndex(first, result));

    bool found = false;
    while (result.position() == position) {
      // first check length to avoid 
      if (meta_len == metadata_len && strncmp(meta_str, metadata, meta_len) == 0) {
        found = true;
        break;
      }
      first++;
      TF_RETURN_IF_ERROR(metadata_->GetRecordAt(first, &meta_str, &meta_len));
      TF_RETURN_IF_ERROR(GetResultAtIndex(first, result));
    }
    
    if (!found)
      return NotFound("Location ", position.DebugString(), " was not found.");
   
    if (index)
      *index = first;

    //result.CopyFrom(result);
    return Status::OK();
  }

    Status AGDResultReader::GetNextResult(Alignment& result) {
      const char* data;
      size_t len;
      TF_RETURN_IF_ERROR(GetNextRecord(&data, &len));
      if (len == 0) {
        return errors::Unavailable("Empty result");
      }
      result.ParseFromArray(data, len);
      return Status::OK();
    }

    Status AGDResultReader::PeekNextResult(Alignment& result) {
      const char* data;
      size_t len;
      TF_RETURN_IF_ERROR(PeekNextRecord(&data, &len));
      if (len == 0) {
        return errors::Unavailable("Empty result");
      }
      result.ParseFromArray(data, len);
      return Status::OK();
    }


    Status AGDResultReader::GetResultAtIndex(size_t index, Alignment& result) {
      const char* data;
      size_t len;
      TF_RETURN_IF_ERROR(GetRecordAt(index, &data, &len));
      if (len == 0) {
        return errors::Unavailable("Empty result");
      }
      result.ParseFromArray(data, len);
      return Status::OK();
    }


}
