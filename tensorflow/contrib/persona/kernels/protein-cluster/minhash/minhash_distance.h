#include "sketch.h"

namespace mash{

	class minhash_distance
	{
	public:
	struct CompareInput
    {
        CompareInput(const Sketch & sketchRefNew, const Sketch & sketchQueryNew, uint64_t indexRefNew, uint64_t indexQueryNew, uint64_t pairCountNew, const Sketch::Parameters & parametersNew, double maxDistanceNew, double maxPValueNew)
            :
            sketchRef(sketchRefNew),
            sketchQuery(sketchQueryNew),
            indexRef(indexRefNew),
            indexQuery(indexQueryNew),
            pairCount(pairCountNew),
            parameters(parametersNew),
            maxDistance(maxDistanceNew),
            maxPValue(maxPValueNew)
            {}
        
        const Sketch & sketchRef;
        const Sketch & sketchQuery;
        
        uint64_t indexRef;
        uint64_t indexQuery;
        uint64_t pairCount;
        
        const Sketch::Parameters & parameters;
        double maxDistance;
        double maxPValue;
    };
    
    struct CompareOutput
    {
        CompareOutput(const Sketch & sketchRefNew, const Sketch & sketchQueryNew, uint64_t indexRefNew, uint64_t indexQueryNew, uint64_t pairCountNew)
            :
            sketchRef(sketchRefNew),
            sketchQuery(sketchQueryNew),
            indexRef(indexRefNew),
            indexQuery(indexQueryNew),
            pairCount(pairCountNew)
        {
            pairs = new PairOutput[pairCount];
        }
        
        ~CompareOutput()
        {
            delete [] pairs;
        }
        
        struct PairOutput
        {
            uint64_t numer;
            uint64_t denom;
            double distance;
            double pValue;
            bool pass;
        };
        
        const Sketch & sketchRef;
        const Sketch & sketchQuery;
        
        uint64_t indexRef;
        uint64_t indexQuery;
        uint64_t pairCount;
        
        PairOutput * pairs;
    };
        
    minhash_distance::CompareOutput * run( char* seqref, char* seqlen, int lengthref, int lengthqry, const Sketch::Parameters & parametersNew); // override
    minhash_distance::CompareOutput * run_seqsketch( Sketch sketchRef,  char* seqqry, int lengthref, int lengthqry, const Sketch::Parameters & parametersNew);
    minhash_distance::CompareOutput * run_seqsketch_repsketch(  Sketch sketchRef, Sketch sketchQry,int lengthref, int lengthqry,const Sketch::Parameters & parametersNew);

    
private:
    
    void writeOutput(CompareOutput * output, bool table) const;
};

minhash_distance::CompareOutput * compare(minhash_distance::CompareInput * input);
void compareSketches(minhash_distance::CompareOutput::PairOutput * output, const Sketch::Reference & refRef, const Sketch::Reference & refQry, uint64_t sketchSize, int kmerSize, double kmerSpace, double maxDistance, double maxPValue);
double pValue(uint64_t x, uint64_t lengthRef, uint64_t lengthQuery, double kmerSpace, uint64_t sketchSize);

}//namespace ends here
