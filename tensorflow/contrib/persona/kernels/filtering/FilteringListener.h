
	#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
	#include <iostream>
	#include <stdlib.h>
	#include <string.h>
	using namespace std;


// Generated from Filtering.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"
#include "FilteringParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by FilteringParser.
 */
class  FilteringListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterProg(FilteringParser::ProgContext *ctx) = 0;
  virtual void exitProg(FilteringParser::ProgContext *ctx) = 0;

  virtual void enterExpression(FilteringParser::ExpressionContext *ctx) = 0;
  virtual void exitExpression(FilteringParser::ExpressionContext *ctx) = 0;

  virtual void enterValue(FilteringParser::ValueContext *ctx) = 0;
  virtual void exitValue(FilteringParser::ValueContext *ctx) = 0;

  virtual void enterIdentifier(FilteringParser::IdentifierContext *ctx) = 0;
  virtual void exitIdentifier(FilteringParser::IdentifierContext *ctx) = 0;


};

