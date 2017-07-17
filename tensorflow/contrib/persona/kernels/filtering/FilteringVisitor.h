
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
 * This class defines an abstract visitor for a parse tree
 * produced by FilteringParser.
 */
class  FilteringVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by FilteringParser.
   */
    virtual antlrcpp::Any visitProg(FilteringParser::ProgContext *context) = 0;

    virtual antlrcpp::Any visitExpression(FilteringParser::ExpressionContext *context) = 0;

    virtual antlrcpp::Any visitValue(FilteringParser::ValueContext *context) = 0;

    virtual antlrcpp::Any visitIdentifier(FilteringParser::IdentifierContext *context) = 0;


};

