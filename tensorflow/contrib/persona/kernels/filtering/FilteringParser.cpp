
	#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
	#include <iostream>
	using namespace std;


// Generated from Filtering.g4 by ANTLR 4.7


#include "FilteringListener.h"
#include "FilteringVisitor.h"

#include "FilteringParser.h"


using namespace antlrcpp;
using namespace antlr4;

FilteringParser::FilteringParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

FilteringParser::~FilteringParser() {
  delete _interpreter;
}

std::string FilteringParser::getGrammarFileName() const {
  return "Filtering.g4";
}

const std::vector<std::string>& FilteringParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& FilteringParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- ProgContext ------------------------------------------------------------------

FilteringParser::ProgContext::ProgContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FilteringParser::ProgContext::EOF() {
  return getToken(FilteringParser::EOF, 0);
}

FilteringParser::ExpressionContext* FilteringParser::ProgContext::expression() {
  return getRuleContext<FilteringParser::ExpressionContext>(0);
}


size_t FilteringParser::ProgContext::getRuleIndex() const {
  return FilteringParser::RuleProg;
}

void FilteringParser::ProgContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FilteringListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProg(this);
}

void FilteringParser::ProgContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FilteringListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProg(this);
}


antlrcpp::Any FilteringParser::ProgContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FilteringVisitor*>(visitor))
    return parserVisitor->visitProg(this);
  else
    return visitor->visitChildren(this);
}

FilteringParser::ProgContext* FilteringParser::prog() {
  ProgContext *_localctx = _tracker.createInstance<ProgContext>(_ctx, getState());
  enterRule(_localctx, 0, FilteringParser::RuleProg);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(8);
    dynamic_cast<ProgContext *>(_localctx)->e = expression(0);
    setState(9);
    match(FilteringParser::EOF);
     cout<<dynamic_cast<ProgContext *>(_localctx)->e->v<<endl ;
    							answer = dynamic_cast<ProgContext *>(_localctx)->e->v ; 
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

FilteringParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FilteringParser::ExpressionContext::LPAREN() {
  return getToken(FilteringParser::LPAREN, 0);
}

tree::TerminalNode* FilteringParser::ExpressionContext::RPAREN() {
  return getToken(FilteringParser::RPAREN, 0);
}

std::vector<FilteringParser::ExpressionContext *> FilteringParser::ExpressionContext::expression() {
  return getRuleContexts<FilteringParser::ExpressionContext>();
}

FilteringParser::ExpressionContext* FilteringParser::ExpressionContext::expression(size_t i) {
  return getRuleContext<FilteringParser::ExpressionContext>(i);
}

tree::TerminalNode* FilteringParser::ExpressionContext::NOT() {
  return getToken(FilteringParser::NOT, 0);
}

std::vector<FilteringParser::ValueContext *> FilteringParser::ExpressionContext::value() {
  return getRuleContexts<FilteringParser::ValueContext>();
}

FilteringParser::ValueContext* FilteringParser::ExpressionContext::value(size_t i) {
  return getRuleContext<FilteringParser::ValueContext>(i);
}

tree::TerminalNode* FilteringParser::ExpressionContext::GT() {
  return getToken(FilteringParser::GT, 0);
}

tree::TerminalNode* FilteringParser::ExpressionContext::GE() {
  return getToken(FilteringParser::GE, 0);
}

tree::TerminalNode* FilteringParser::ExpressionContext::LT() {
  return getToken(FilteringParser::LT, 0);
}

tree::TerminalNode* FilteringParser::ExpressionContext::LE() {
  return getToken(FilteringParser::LE, 0);
}

tree::TerminalNode* FilteringParser::ExpressionContext::EQ() {
  return getToken(FilteringParser::EQ, 0);
}

tree::TerminalNode* FilteringParser::ExpressionContext::NE() {
  return getToken(FilteringParser::NE, 0);
}

tree::TerminalNode* FilteringParser::ExpressionContext::AND() {
  return getToken(FilteringParser::AND, 0);
}

tree::TerminalNode* FilteringParser::ExpressionContext::OR() {
  return getToken(FilteringParser::OR, 0);
}


size_t FilteringParser::ExpressionContext::getRuleIndex() const {
  return FilteringParser::RuleExpression;
}

void FilteringParser::ExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FilteringListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpression(this);
}

void FilteringParser::ExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FilteringListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpression(this);
}


antlrcpp::Any FilteringParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FilteringVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}


FilteringParser::ExpressionContext* FilteringParser::expression() {
   return expression(0);
}

FilteringParser::ExpressionContext* FilteringParser::expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  FilteringParser::ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, parentState);
  FilteringParser::ExpressionContext *previousContext = _localctx;
  size_t startState = 2;
  enterRecursionRule(_localctx, 2, FilteringParser::RuleExpression, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(27);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FilteringParser::LPAREN: {
        setState(13);
        match(FilteringParser::LPAREN);
        setState(14);
        dynamic_cast<ExpressionContext *>(_localctx)->e = expression(0);
        setState(15);
        match(FilteringParser::RPAREN);
         dynamic_cast<ExpressionContext *>(_localctx)->v =  dynamic_cast<ExpressionContext *>(_localctx)->e->v ; 
        break;
      }

      case FilteringParser::NOT: {
        setState(18);
        match(FilteringParser::NOT);
        setState(19);
        dynamic_cast<ExpressionContext *>(_localctx)->e = expression(3);
         dynamic_cast<ExpressionContext *>(_localctx)->v =  !(dynamic_cast<ExpressionContext *>(_localctx)->e->v) ; 
        break;
      }

      case FilteringParser::RESULT:
      case FilteringParser::Number: {
        setState(22);
        dynamic_cast<ExpressionContext *>(_localctx)->a = value();
        setState(23);
        dynamic_cast<ExpressionContext *>(_localctx)->c = _input->LT(1);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << FilteringParser::GT)
          | (1ULL << FilteringParser::GE)
          | (1ULL << FilteringParser::LT)
          | (1ULL << FilteringParser::LE)
          | (1ULL << FilteringParser::EQ)
          | (1ULL << FilteringParser::NE))) != 0))) {
          dynamic_cast<ExpressionContext *>(_localctx)->c = _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(24);
        dynamic_cast<ExpressionContext *>(_localctx)->b = value();
         
        									  if( dynamic_cast<ExpressionContext *>(_localctx)->c->getType() == GT )
        										dynamic_cast<ExpressionContext *>(_localctx)->v =  (dynamic_cast<ExpressionContext *>(_localctx)->a->n > dynamic_cast<ExpressionContext *>(_localctx)->b->n) ; 
        									  else if( dynamic_cast<ExpressionContext *>(_localctx)->c->getType() == GE )
        										dynamic_cast<ExpressionContext *>(_localctx)->v =  (dynamic_cast<ExpressionContext *>(_localctx)->a->n >= dynamic_cast<ExpressionContext *>(_localctx)->b->n) ;
        									  else if( dynamic_cast<ExpressionContext *>(_localctx)->c->getType() == LT )
        										dynamic_cast<ExpressionContext *>(_localctx)->v =  (dynamic_cast<ExpressionContext *>(_localctx)->a->n < dynamic_cast<ExpressionContext *>(_localctx)->b->n) ;
        									  else if( dynamic_cast<ExpressionContext *>(_localctx)->c->getType() == LE )
        										dynamic_cast<ExpressionContext *>(_localctx)->v =  (dynamic_cast<ExpressionContext *>(_localctx)->a->n <= dynamic_cast<ExpressionContext *>(_localctx)->b->n) ;
        									  else if( dynamic_cast<ExpressionContext *>(_localctx)->c->getType() == EQ )
        										dynamic_cast<ExpressionContext *>(_localctx)->v =  (dynamic_cast<ExpressionContext *>(_localctx)->a->n == dynamic_cast<ExpressionContext *>(_localctx)->b->n) ;
        									  else if( dynamic_cast<ExpressionContext *>(_localctx)->c->getType() == NE )
        										dynamic_cast<ExpressionContext *>(_localctx)->v =  (dynamic_cast<ExpressionContext *>(_localctx)->a->n != dynamic_cast<ExpressionContext *>(_localctx)->b->n) ;
        									
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(36);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ExpressionContext>(parentContext, parentState);
        _localctx->x = previousContext;
        pushNewRecursionContext(_localctx, startState, RuleExpression);
        setState(29);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(30);
        dynamic_cast<ExpressionContext *>(_localctx)->o = _input->LT(1);
        _la = _input->LA(1);
        if (!(_la == FilteringParser::AND

        || _la == FilteringParser::OR)) {
          dynamic_cast<ExpressionContext *>(_localctx)->o = _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(31);
        dynamic_cast<ExpressionContext *>(_localctx)->y = expression(2);
         
                  											  if( dynamic_cast<ExpressionContext *>(_localctx)->o->getType() == AND )
                  													dynamic_cast<ExpressionContext *>(_localctx)->v =  (dynamic_cast<ExpressionContext *>(_localctx)->x->v && dynamic_cast<ExpressionContext *>(_localctx)->y->v) ;
                  											  else  dynamic_cast<ExpressionContext *>(_localctx)->v =  (dynamic_cast<ExpressionContext *>(_localctx)->x->v || dynamic_cast<ExpressionContext *>(_localctx)->y->v) ; 
                  											 
      }
      setState(38);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ValueContext ------------------------------------------------------------------

FilteringParser::ValueContext::ValueContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FilteringParser::ValueContext::Number() {
  return getToken(FilteringParser::Number, 0);
}

FilteringParser::IdentifierContext* FilteringParser::ValueContext::identifier() {
  return getRuleContext<FilteringParser::IdentifierContext>(0);
}


size_t FilteringParser::ValueContext::getRuleIndex() const {
  return FilteringParser::RuleValue;
}

void FilteringParser::ValueContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FilteringListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterValue(this);
}

void FilteringParser::ValueContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FilteringListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitValue(this);
}


antlrcpp::Any FilteringParser::ValueContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FilteringVisitor*>(visitor))
    return parserVisitor->visitValue(this);
  else
    return visitor->visitChildren(this);
}

FilteringParser::ValueContext* FilteringParser::value() {
  ValueContext *_localctx = _tracker.createInstance<ValueContext>(_ctx, getState());
  enterRule(_localctx, 4, FilteringParser::RuleValue);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(44);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FilteringParser::Number: {
        enterOuterAlt(_localctx, 1);
        setState(39);
        dynamic_cast<ValueContext *>(_localctx)->numberToken = match(FilteringParser::Number);
         dynamic_cast<ValueContext *>(_localctx)->n =  stoi(dynamic_cast<ValueContext *>(_localctx)->numberToken->getText()) ; 
        break;
      }

      case FilteringParser::RESULT: {
        enterOuterAlt(_localctx, 2);
        setState(41);
        dynamic_cast<ValueContext *>(_localctx)->identifierContext = identifier();
         dynamic_cast<ValueContext *>(_localctx)->n =  dynamic_cast<ValueContext *>(_localctx)->identifierContext->n ; 
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdentifierContext ------------------------------------------------------------------

FilteringParser::IdentifierContext::IdentifierContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FilteringParser::IdentifierContext::RESULT() {
  return getToken(FilteringParser::RESULT, 0);
}

tree::TerminalNode* FilteringParser::IdentifierContext::DOT() {
  return getToken(FilteringParser::DOT, 0);
}

tree::TerminalNode* FilteringParser::IdentifierContext::FLAG() {
  return getToken(FilteringParser::FLAG, 0);
}


size_t FilteringParser::IdentifierContext::getRuleIndex() const {
  return FilteringParser::RuleIdentifier;
}

void FilteringParser::IdentifierContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FilteringListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdentifier(this);
}

void FilteringParser::IdentifierContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FilteringListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdentifier(this);
}


antlrcpp::Any FilteringParser::IdentifierContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FilteringVisitor*>(visitor))
    return parserVisitor->visitIdentifier(this);
  else
    return visitor->visitChildren(this);
}

FilteringParser::IdentifierContext* FilteringParser::identifier() {
  IdentifierContext *_localctx = _tracker.createInstance<IdentifierContext>(_ctx, getState());
  enterRule(_localctx, 6, FilteringParser::RuleIdentifier);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(46);
    match(FilteringParser::RESULT);
    setState(47);
    match(FilteringParser::DOT);
    setState(48);
    match(FilteringParser::FLAG);
     dynamic_cast<IdentifierContext *>(_localctx)->n =  result.flag() ; 
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool FilteringParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 1: return expressionSempred(dynamic_cast<ExpressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool FilteringParser::expressionSempred(ExpressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> FilteringParser::_decisionToDFA;
atn::PredictionContextCache FilteringParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN FilteringParser::_atn;
std::vector<uint16_t> FilteringParser::_serializedATN;

std::vector<std::string> FilteringParser::_ruleNames = {
  "prog", "expression", "value", "identifier"
};

std::vector<std::string> FilteringParser::_literalNames = {
  "", "'result'", "'mate'", "'flag'", "'mapq'", "'position'", "'('", "')'", 
  "", "", "'NOT'", "'TRUE'", "'FALSE'", "'>'", "'>='", "'<'", "'<='", "'=='", 
  "'!='", "'.'"
};

std::vector<std::string> FilteringParser::_symbolicNames = {
  "", "RESULT", "MATE", "FLAG", "MAPQ", "POSITION", "LPAREN", "RPAREN", 
  "AND", "OR", "NOT", "TRUE", "FALSE", "GT", "GE", "LT", "LE", "EQ", "NE", 
  "DOT", "Number", "WS"
};

dfa::Vocabulary FilteringParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> FilteringParser::_tokenNames;

FilteringParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x17, 0x36, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 0x9, 
    0x4, 0x4, 0x5, 0x9, 0x5, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x5, 0x3, 0x1e, 0xa, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x7, 0x3, 0x25, 0xa, 0x3, 0xc, 0x3, 0xe, 0x3, 0x28, 0xb, 0x3, 
    0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x5, 0x4, 0x2f, 0xa, 
    0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x2, 
    0x3, 0x4, 0x6, 0x2, 0x4, 0x6, 0x8, 0x2, 0x4, 0x3, 0x2, 0xf, 0x14, 0x3, 
    0x2, 0xa, 0xb, 0x2, 0x35, 0x2, 0xa, 0x3, 0x2, 0x2, 0x2, 0x4, 0x1d, 0x3, 
    0x2, 0x2, 0x2, 0x6, 0x2e, 0x3, 0x2, 0x2, 0x2, 0x8, 0x30, 0x3, 0x2, 0x2, 
    0x2, 0xa, 0xb, 0x5, 0x4, 0x3, 0x2, 0xb, 0xc, 0x7, 0x2, 0x2, 0x3, 0xc, 
    0xd, 0x8, 0x2, 0x1, 0x2, 0xd, 0x3, 0x3, 0x2, 0x2, 0x2, 0xe, 0xf, 0x8, 
    0x3, 0x1, 0x2, 0xf, 0x10, 0x7, 0x8, 0x2, 0x2, 0x10, 0x11, 0x5, 0x4, 
    0x3, 0x2, 0x11, 0x12, 0x7, 0x9, 0x2, 0x2, 0x12, 0x13, 0x8, 0x3, 0x1, 
    0x2, 0x13, 0x1e, 0x3, 0x2, 0x2, 0x2, 0x14, 0x15, 0x7, 0xc, 0x2, 0x2, 
    0x15, 0x16, 0x5, 0x4, 0x3, 0x5, 0x16, 0x17, 0x8, 0x3, 0x1, 0x2, 0x17, 
    0x1e, 0x3, 0x2, 0x2, 0x2, 0x18, 0x19, 0x5, 0x6, 0x4, 0x2, 0x19, 0x1a, 
    0x9, 0x2, 0x2, 0x2, 0x1a, 0x1b, 0x5, 0x6, 0x4, 0x2, 0x1b, 0x1c, 0x8, 
    0x3, 0x1, 0x2, 0x1c, 0x1e, 0x3, 0x2, 0x2, 0x2, 0x1d, 0xe, 0x3, 0x2, 
    0x2, 0x2, 0x1d, 0x14, 0x3, 0x2, 0x2, 0x2, 0x1d, 0x18, 0x3, 0x2, 0x2, 
    0x2, 0x1e, 0x26, 0x3, 0x2, 0x2, 0x2, 0x1f, 0x20, 0xc, 0x3, 0x2, 0x2, 
    0x20, 0x21, 0x9, 0x3, 0x2, 0x2, 0x21, 0x22, 0x5, 0x4, 0x3, 0x4, 0x22, 
    0x23, 0x8, 0x3, 0x1, 0x2, 0x23, 0x25, 0x3, 0x2, 0x2, 0x2, 0x24, 0x1f, 
    0x3, 0x2, 0x2, 0x2, 0x25, 0x28, 0x3, 0x2, 0x2, 0x2, 0x26, 0x24, 0x3, 
    0x2, 0x2, 0x2, 0x26, 0x27, 0x3, 0x2, 0x2, 0x2, 0x27, 0x5, 0x3, 0x2, 
    0x2, 0x2, 0x28, 0x26, 0x3, 0x2, 0x2, 0x2, 0x29, 0x2a, 0x7, 0x16, 0x2, 
    0x2, 0x2a, 0x2f, 0x8, 0x4, 0x1, 0x2, 0x2b, 0x2c, 0x5, 0x8, 0x5, 0x2, 
    0x2c, 0x2d, 0x8, 0x4, 0x1, 0x2, 0x2d, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x2e, 
    0x29, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x2f, 0x7, 
    0x3, 0x2, 0x2, 0x2, 0x30, 0x31, 0x7, 0x3, 0x2, 0x2, 0x31, 0x32, 0x7, 
    0x15, 0x2, 0x2, 0x32, 0x33, 0x7, 0x5, 0x2, 0x2, 0x33, 0x34, 0x8, 0x5, 
    0x1, 0x2, 0x34, 0x9, 0x3, 0x2, 0x2, 0x2, 0x5, 0x1d, 0x26, 0x2e, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

FilteringParser::Initializer FilteringParser::_init;
