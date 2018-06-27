
#ifndef EXTRAS_H_
#define EXTRAS_H_

#include "matrix.h"
#include "DynProgr_sse_byte.h"
#include "DynProgr_sse_short.h"

#define MAXSEQLEN    100010
#define MINUSINF (-999999999)
#define MAXMUTDIM       130

#define MMAX(a,b) ((a)>(b)?(a):(b))

//extern double coldel[MAXSEQLEN+1], S[MAXSEQLEN+1];
//extern int DelFrom[MAXSEQLEN+1];
typedef struct {
  double coldel[MAXSEQLEN+1], S[MAXSEQLEN+1];
  int DelFrom[MAXSEQLEN+1];
} BTData;

#endif
