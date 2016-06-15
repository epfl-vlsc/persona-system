#include "decompress.h"
#include "tensorflow/core/platform/logging.h"
#include <zlib.h>

namespace tensorflow {

using namespace std;
using namespace errors;

namespace {

static const int ENABLE_ZLIB_GZIP = 32;
static const int window_bits = 15;
static const size_t extend_length = 1024 * 1024 * 2; // 2 Mb

Status resize_output(z_stream &strm, vector<char> &output, size_t extend_len) {
  auto s = Status::OK();

  if (strm.avail_out == 0) {
    auto new_cap = output.capacity() + extend_len;
    output.reserve(new_cap);
    if (output.capacity() < new_cap) {
      s = Internal("Unable to reserve more capacity in a buffer");
    } else {
      strm.next_out = reinterpret_cast<unsigned char*>(&output[strm.total_out]);
      // should never be negative
      strm.avail_out = output.capacity() - strm.total_out;
    }
  }

  return s;
}

}

Status decompressGZIP(const char* segment,
                      const size_t segment_size,
                      vector<char> &output)
{
  // TODO this only supports decompress write, not appending
  // this is an easy change to make, but requires some more "math"
  output.clear(); // just to be sure, in case the caller didn't do it
  static const int init_flags = window_bits | ENABLE_ZLIB_GZIP;
  z_stream strm = {0};
  strm.zalloc = Z_NULL;
  strm.zfree = Z_NULL;
  strm.next_in = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(segment));
  strm.avail_in = segment_size;

  int status = inflateInit2(&strm, init_flags);
  if (status != Z_OK) {
    return Internal("inflateInit2 failed with error: ", status);
  }

  auto s = Status::OK();

  // First, try to decompress as much as possible in a single step
  strm.avail_out = output.capacity();
  strm.next_out = reinterpret_cast<unsigned char*>(&output[0]);
  status = inflate(&strm, Z_FINISH);

  if (status != Z_STREAM_END) {
    if (status == Z_MEM_ERROR || status == Z_BUF_ERROR) {
      // Do normal decompression because we couldn't do it in one shot
      do {
        s = resize_output(strm, output, extend_length);
        if (!s.ok()) {
          break;
        }

        status = inflate(&strm, Z_NO_FLUSH);
        switch (status) {
        case Z_OK:
        case Z_STREAM_END:
          break;
        default: // an error
          s = Internal("inflate(Z_NO_FLUSH) returned code ", status, " with message '", strm.msg == NULL ? "" : strm.msg, "'");
          break;
        }
      } while (status != Z_STREAM_END && s.ok());
    } else {
      s = Internal("inflate(Z_FINISH) return code ", status, " with message '", strm.msg == NULL ? "" : strm.msg, "'");
    }
  }

  auto total_out = strm.total_out;

  status = inflateEnd(&strm);
  if (s.ok() && status != Z_OK) { // s.ok() status to make sure we don't override non-inflateEnd error
    s = Internal("inflateEnd() didn't receive Z_OK. Got: ", status);
    output.clear();
  } else {
    output.resize(total_out);
  }
  return s;
}

} // namespace tensorflow
