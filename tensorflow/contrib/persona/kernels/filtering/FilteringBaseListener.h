
	#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
	#include <iostream>
	#include <stdlib.h>
	#include <string.h>
	using namespace std;


// Generated from Filtering.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"
#include "FilteringListener.h"


/**
 * This class provides an empty implementation of FilteringListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  FilteringBaseListener : public FilteringListener {
public:

  virtual void enterProg(FilteringParser::ProgContext * /*ctx*/) override { }
  virtual void exitProg(FilteringParser::ProgContext * /*ctx*/) override { }

  virtual void enterExpression(FilteringParser::ExpressionContext * /*ctx*/) override { }
  virtual void exitExpression(FilteringParser::ExpressionContext * /*ctx*/) override { }

  virtual void enterValue(FilteringParser::ValueContext * /*ctx*/) override { }
  virtual void exitValue(FilteringParser::ValueContext * /*ctx*/) override { }

  virtual void enterIdentifier(FilteringParser::IdentifierContext * /*ctx*/) override { }
  virtual void exitIdentifier(FilteringParser::IdentifierContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

