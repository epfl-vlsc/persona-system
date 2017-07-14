
	#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
	#include <iostream>
	using namespace std;


// Generated from Filtering.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"
#include "FilteringVisitor.h"


/**
 * This class provides an empty implementation of FilteringVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  FilteringBaseVisitor : public FilteringVisitor {
public:

  virtual antlrcpp::Any visitProg(FilteringParser::ProgContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpression(FilteringParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitValue(FilteringParser::ValueContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIdentifier(FilteringParser::IdentifierContext *ctx) override {
    return visitChildren(ctx);
  }


};

