# To regenerate ANTLR files, run the following command in .../kernels/filtering/
	"java -jar antlr-4.7-complete.jar -Dlanguage=Cpp -listener -visitor Filtering.g4"

# To rebuild ANTLR library, clone github repo of ANTLR4 and look for instructions for the C++ target.
https://github.com/antlr/antlr4/tree/master/runtime/Cpp#compiling-on-linux

	1. cd /runtime/Cpp (this is where this readme is located)
	2. mkdir build && mkdir run && cd build
	3. cmake .. -DANTLR_JAR_LOCATION=full/path/to/antlr-4-7-complete.jar
	4. make
	5. DESTDIR=/runtime/Cpp/run make install

Chose the DESTDIR as required.
Before running make (Step 4), to the CMakelists file add the following line to use the fPIC option

	"set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )""

# While compiling the Op, I have used -rpath option to give runtime environment. This is an alternate to adding the library path in $LD_LIBRARY_PATH.

# The grammar allows queries on the following parameters :
	 result.flag
	 result.mapq
	 result.ref_index
	 result.position
	 mate.position
	 mate.ref_index

	 It allows the following comparisons with integers : >,>=,<,<=,==,!=
	 It allows bitwise and with Hexadecimal numbers (and integers too) also, which is generally required for result flags : result.flag & 0x0010 != 0
	 It allows combinations of boolean results using AND, OR and NOT.
	 ( result.flag & 0x10 != 0 ) AND (result.mapq >= 50)

# Errors :
	1. Issue with last chunk read.
	2. Sometimes, running with chunk size 2 gives the error 
		 F ./tensorflow/core/lib/core/refcount.h:87] Check failed: ref_.load() > 0 (0 vs. 0)
	3. Very very rarely, gives 'python corrupted double linked list error' (with any parameters).