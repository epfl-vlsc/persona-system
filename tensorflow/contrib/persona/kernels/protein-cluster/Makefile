CC = g++
CFLAGS := -Wall -pipe -O2
CXXFLAGS := $(CFLAGS)
LOBJS = ssw.o
PROG = ssw_test
LIB = libssw.so

.PHONY: all clean

all: $(PROG)  $(LIB) 

$(LIB): ssw.cpp ssw.h
	$(CC) $(CFLAGS) -fPIC -shared -rdynamic -o $@ $<

$(PROG): main.cpp kseq.h

$(PROG) $(EXAMPLE): $(LOBJS)
	$(CC) -o $@ $(filter-out %.h,$^) $(CFLAGS) -lm -lz

ssw.o: ssw.cpp ssw.h
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	-rm -f $(LOBJS) $(PROG) $(LIB) $(EXAMPLE) *~ 
