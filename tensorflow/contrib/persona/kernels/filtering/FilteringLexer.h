
// Generated from Filtering.g4 by ANTLR 4.7

#pragma once


#include "antlr4-runtime.h"




class  FilteringLexer : public antlr4::Lexer {
public:
  enum {
    RESULT = 1, MATE = 2, FLAG = 3, MAPQ = 4, POSITION = 5, REF_INDEX = 6, 
    LPAREN = 7, RPAREN = 8, AND = 9, OR = 10, NOT = 11, TRUE = 12, FALSE = 13, 
    GT = 14, GE = 15, LT = 16, LE = 17, EQ = 18, NE = 19, BITAND = 20, DOT = 21, 
    Dec_Number = 22, Hex_Number = 23, WS = 24
  };

  FilteringLexer(antlr4::CharStream *input);
  ~FilteringLexer();

  virtual std::string getGrammarFileName() const override;
  virtual const std::vector<std::string>& getRuleNames() const override;

  virtual const std::vector<std::string>& getChannelNames() const override;
  virtual const std::vector<std::string>& getModeNames() const override;
  virtual const std::vector<std::string>& getTokenNames() const override; // deprecated, use vocabulary instead
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;

  virtual const std::vector<uint16_t> getSerializedATN() const override;
  virtual const antlr4::atn::ATN& getATN() const override;

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;
  static std::vector<std::string> _channelNames;
  static std::vector<std::string> _modeNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

