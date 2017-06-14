/*
* This work uses NCBI SRA API
* @author: Alaleh Azhir
*/

#include <ncbi-vdb/NGS.hpp>
#include <ngs/ErrorMsg.hpp>
#include <ngs/ReadCollection.hpp>
#include <ngs/ReadIterator.hpp>
#include <iostream>
#include <string>

using namespace ngs;
using namespace std;

class ReadSra
{
public:

    static void run ( const String & acc )
    {

        ReadCollection run = ncbi::NGS::openReadCollection ( acc );
	ReadIterator it = run.getReadRange(1, 1);
	bool k = it.nextRead();
	cout << "k is not " << !k << "\n";
        StringRef ReadName = it . getReadName ();
        StringRef bases = it . getReadBases ();
	StringRef quality = it . getReadQualities ();
	cout << "name is: " << ReadName << "\n";
	cout << "bases are " << bases.size() << "\n";
	cout << "qualities are" << quality.size() << "\n";
        cout << "number is" << run.getReadCount() << "\n";  
    }
};

int main (int argc, char const *argv[])
{
    if ( argc != 2 )
    {
        cerr << "Please enter an SRA file\n";
    }
    else try
    {
        ReadSra::run ( argv[1] );
        return 0;
    }
    catch ( ErrorMsg & x )
    {
        cerr <<  x.toString () << '\n';
    }
    catch ( exception & x )
    {
        cerr <<  x.what () << '\n';
    }
    catch ( ... )
    {
        cerr <<  "unknown exception\n";
    }

    return 10;
}
