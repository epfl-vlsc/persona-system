
	#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
	#include <iostream>
	#include <stdlib.h>
	#include <string.h>
	using namespace std;


// Generated from Filtering.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"




class  FilteringParser : public antlr4::Parser {
public:
  enum {
    RESULT = 1, MATE = 2, FLAG = 3, MAPQ = 4, POSITION = 5, REF_INDEX = 6, 
    LPAREN = 7, RPAREN = 8, AND = 9, OR = 10, NOT = 11, TRUE = 12, FALSE = 13, 
    GT = 14, GE = 15, LT = 16, LE = 17, EQ = 18, NE = 19, BITAND = 20, DOT = 21, 
    Dec_Number = 22, Hex_Number = 23, WS = 24
  };

  enum {
    RuleProg = 0, RuleExpression = 1, RuleValue = 2, RuleIdentifier = 3
  };

  FilteringParser(antlr4::TokenStream *input);
  ~FilteringParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


  	Alignment result;
  	bool answer;
  	/*
  	FilteringParser(antlr4::TokenStream *input, int result)	: Parser(input) {
    		//_interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
    		this->FilteringParser(input);
  		this->res_flag = result;
  	}
  	*/
  	FilteringParser(antlr4::TokenStream *input, Alignment res)	: FilteringParser(input) {	// Calling the default constructor from my constructor (constructor-constructor calls allowed in C++11)
  		this->result = res;
  	}


  class ProgContext;
  class ExpressionContext;
  class ValueContext;
  class IdentifierContext; 

  class  ProgContext : public antlr4::ParserRuleContext {
  public:
    FilteringParser::ExpressionContext *e = nullptr;;
    ProgContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EOF();
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ProgContext* prog();

  class  ExpressionContext : public antlr4::ParserRuleContext {
  public:
    bool v;
    FilteringParser::ExpressionContext *x = nullptr;;
    FilteringParser::ExpressionContext *e = nullptr;;
    FilteringParser::ValueContext *a = nullptr;;
    antlr4::Token *c = nullptr;;
    FilteringParser::ValueContext *b = nullptr;;
    antlr4::Token *o = nullptr;;
    FilteringParser::ExpressionContext *y = nullptr;;
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    antlr4::tree::TerminalNode *NOT();
    std::vector<ValueContext *> value();
    ValueContext* value(size_t i);
    antlr4::tree::TerminalNode *GT();
    antlr4::tree::TerminalNode *GE();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *LE();
    antlr4::tree::TerminalNode *EQ();
    antlr4::tree::TerminalNode *NE();
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *OR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionContext* expression();
  ExpressionContext* expression(int precedence);
  class  ValueContext : public antlr4::ParserRuleContext {
  public:
    int n;
    FilteringParser::ValueContext *a = nullptr;;
    FilteringParser::ValueContext *d = nullptr;;
    antlr4::Token *dec_numberToken = nullptr;;
    antlr4::Token *hex_numberToken = nullptr;;
    FilteringParser::IdentifierContext *identifierContext = nullptr;;
    antlr4::Token *c = nullptr;;
    FilteringParser::ValueContext *b = nullptr;;
    ValueContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ValueContext *> value();
    ValueContext* value(size_t i);
    antlr4::tree::TerminalNode *Dec_Number();
    antlr4::tree::TerminalNode *Hex_Number();
    IdentifierContext *identifier();
    antlr4::tree::TerminalNode *BITAND();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ValueContext* value();
  ValueContext* value(int precedence);
  class  IdentifierContext : public antlr4::ParserRuleContext {
  public:
    int n;
    IdentifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RESULT();
    antlr4::tree::TerminalNode *DOT();
    antlr4::tree::TerminalNode *FLAG();
    antlr4::tree::TerminalNode *MAPQ();
    antlr4::tree::TerminalNode *POSITION();
    antlr4::tree::TerminalNode *REF_INDEX();
    antlr4::tree::TerminalNode *MATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IdentifierContext* identifier();


  virtual bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;
  bool expressionSempred(ExpressionContext *_localctx, size_t predicateIndex);
  bool valueSempred(ValueContext *_localctx, size_t predicateIndex);

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

