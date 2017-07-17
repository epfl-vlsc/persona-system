grammar Filtering;

@parser::header
{
	#include "tensorflow/contrib/persona/kernels/agd-format/proto/alignment.pb.h"
	#include <iostream>
	#include <stdlib.h>
	#include <string.h>
	using namespace std;
}

@parser::members 
{
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
}


prog: e=expression EOF 	{ //cout<<$e.v<<endl ;
							answer = $e.v ; }
	; 

expression returns [bool v]
	: LPAREN e=expression RPAREN	{ $v = $e.v ; }
	| NOT e=expression	{ $v = !($e.v) ; }
	| a=value c=(GT|GE|LT|LE|EQ|NE) b=value	{ 
									  if( $c->getType() == GT )
										$v = ($a.n > $b.n) ; 
									  else if( $c->getType() == GE )
										$v = ($a.n >= $b.n) ;
									  else if( $c->getType() == LT )
										$v = ($a.n < $b.n) ;
									  else if( $c->getType() == LE )
										$v = ($a.n <= $b.n) ;
									  else if( $c->getType() == EQ )
										$v = ($a.n == $b.n) ;
									  else if( $c->getType() == NE )
										$v = ($a.n != $b.n) ;
									}
	| x=expression o=(AND|OR) y=expression	{ 
											  if( $o->getType() == AND )
													$v = ($x.v && $y.v) ;
											  else  $v = ($x.v || $y.v) ; 
											}
	;

value returns [int n]
	: LPAREN d=value RPAREN		{ $n = $d.n ; }
	| Dec_Number	{ $n = stoi($Dec_Number->getText()) ; }
	| Hex_Number	{ $n = strtol($Hex_Number->getText().c_str(),NULL,0) ; }
	| a=value c=BITAND b=value	{ $n = ($a.n & $b.n) ; }
	| identifier { $n = $identifier.n ; }
	;

identifier returns [int n]
	: RESULT DOT FLAG 	{ $n = result.flag() ; }
	| RESULT DOT MAPQ 	{ $n = result.mapping_quality() ; }
	| RESULT DOT POSITION { $n = result.position().position() ; }
	| RESULT DOT REF_INDEX { $n = result.position().ref_index() ; }
	| MATE DOT POSITION { $n = result.next_position().position() ; }
	| MATE DOT REF_INDEX { $n = result.next_position().ref_index() ; }
	;
/*	
comparator
	: GT | GE | LT | LE | EQ | NE 
	;

binary_op 
	: AND | OR 
	;
*/
/*
identifier: read DOT parameter
	;

read
	: RESULT 
	| MATE
	;

parameter
	: FLAG
	| MAPQ
	| POSITION
	;
*/
RESULT: 'result' ;
MATE: 'mate' ;
FLAG: 'flag' ;
MAPQ: 'mapq' ;
POSITION: 'position' ;
REF_INDEX: 'ref_index' ;
LPAREN	   : '(' ;
RPAREN	   : ')' ;
AND        : 'AND' | '&&' ;
OR         : 'OR' | '||' ;
NOT        : 'NOT';
TRUE       : 'TRUE' ;
FALSE      : 'FALSE' ;
GT         : '>' ;
GE         : '>=' ;
LT         : '<' ;
LE   	   : '<=' ;
EQ 		   : '==' ;
NE 		   : '!=' ;
BITAND	   : '&' ;
DOT		   : '.' ;

Dec_Number: [0-9]+ 
	;

Hex_Number: ('0x'|'0X')[0-9]+
	;

WS : [ \t\r\n\u000C]+ -> skip
   ;