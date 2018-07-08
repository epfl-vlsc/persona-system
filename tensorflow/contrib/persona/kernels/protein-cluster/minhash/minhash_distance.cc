#include "minhash_distance.h"
#include <math.h>
#include <vector>

#include <iostream>
#ifdef USE_BOOST
    #include <boost/math/distributions/binomial.hpp>
    using namespace::boost::math;
#else
    #include <gsl/gsl_cdf.h>
#endif

using namespace::std;

namespace mash {


minhash_distance::CompareOutput * compare(minhash_distance::CompareInput * input)
{
    const Sketch & sketchRef = input->sketchRef;
    const Sketch & sketchQuery = input->sketchQuery;
    
    minhash_distance::CompareOutput * output = new minhash_distance::CompareOutput(input->sketchRef, input->sketchQuery, input->indexRef, input->indexQuery, input->pairCount);
    
    uint64_t sketchSize = sketchQuery.getMinHashesPerWindow() < sketchRef.getMinHashesPerWindow() ?
        sketchQuery.getMinHashesPerWindow() :
        sketchRef.getMinHashesPerWindow();
    
    uint64_t i = input->indexQuery;
    uint64_t j = input->indexRef;
    
    for ( uint64_t k = 0; k < input->pairCount && i < sketchQuery.getReferenceCount(); k++ )
    {
        compareSketches(&output->pairs[k], sketchRef.getReference(j), sketchQuery.getReference(i), sketchSize, sketchRef.getKmerSize(), sketchRef.getKmerSpace(), input->maxDistance, input->maxPValue);
        
        j++;
        
        if ( j == sketchRef.getReferenceCount() )
        {
            j = 0;
            i++;
        }
    }
    
    return output;
}


double pValue(uint64_t x, uint64_t lengthRef, uint64_t lengthQuery, double kmerSpace, uint64_t sketchSize)
{
    if ( x == 0 )
    {
        return 1.;
    }
    
    double pX = 1. / (1. + kmerSpace / lengthRef);
    double pY = 1. / (1. + kmerSpace / lengthQuery);
    
    double r = pX * pY / (pX + pY - pX * pY);
    
    // double M = (double)kmerSpace * (pX + pY) / (1. + r);
    
    // return gsl_cdf_hypergeometric_Q(x - 1, r * M, M - r * M, sketchSize);
    
#ifdef USE_BOOST
    return cdf(complement(binomial(sketchSize, r), x - 1));
#else
    return gsl_cdf_binomial_Q(x - 1, r, sketchSize);
#endif
}

void compareSketches(minhash_distance::CompareOutput::PairOutput * output, const Sketch::Reference & refRef, const Sketch::Reference & refQry, uint64_t sketchSize, int kmerSize, double kmerSpace, double maxDistance, double maxPValue)
{
    uint64_t i = 0;
    uint64_t j = 0;
    uint64_t common = 0;
    uint64_t denom = 0;
    const HashList & hashesSortedRef = refRef.hashesSorted;
    const HashList & hashesSortedQry = refQry.hashesSorted;
    
    output->pass = false;
    
    while ( denom < sketchSize && i < hashesSortedRef.size() && j < hashesSortedQry.size() )
    {
        if ( hashLessThan(hashesSortedRef.at(i), hashesSortedQry.at(j), hashesSortedRef.get64()) )
        {
            i++;
        }
        else if ( hashLessThan(hashesSortedQry.at(j), hashesSortedRef.at(i), hashesSortedRef.get64()) )
        {
            j++;
        }
        else
        {
            i++;
            j++;
            common++;
        }
        
        denom++;
    }
    
    if ( denom < sketchSize )
    {
        // complete the union operation if possible
        
        if ( i < hashesSortedRef.size() )
        {
            denom += hashesSortedRef.size() - i;
        }
        
        if ( j < hashesSortedQry.size() )
        {
            denom += hashesSortedQry.size() - j;
        }
        
        if ( denom > sketchSize )
        {
            denom = sketchSize;
        }
    }
    
    double distance;
    double jaccard = double(common) / denom;
    
    if ( common == denom ) // avoid -0
    {
        distance = 0;
    }
    else if ( common == 0 ) // avoid inf
    {
        distance = 1.;
    }
    else
    {
        //distance = log(double(common + 1) / (denom + 1)) / log(1. / (denom + 1));
        distance = -log(2 * jaccard / (1. + jaccard)) / kmerSize;
    }
    // std::cout << "Max distance is " << maxDistance << endl;
    // if ( distance > maxDistance )
    // {
    //     return;
    // }
    
    output->numer = common;
    output->denom = denom;
    output->distance = distance;
    output->pValue = pValue(common, refRef.length, refQry.length, kmerSpace, denom);
    
    if ( output->pValue > maxPValue )
    {
        return;
    }
    
    output->pass = true;
}



minhash_distance::CompareOutput * minhash_distance::run( char* seqref,  char*seqqry, int lengthref, int lengthqry, const Sketch::Parameters & parametersNew)
{
    //we need only the seqNew, lengthNew, and parametersNew to be sent. Remaining all are empty strings.

 	Sketch sketchRef;
    
    uint64_t lengthMax;
    double randomChance;
    int kMin;
    string lengthMaxName;
    int warningCount = 0;


    sketchRef.init( seqref, lengthref, "", "", parametersNew);
    Sketch sketchQuery;
    sketchQuery.init( seqqry, lengthqry, "", "", parametersNew);

    uint64_t pairCount = sketchRef.getReferenceCount() * sketchQuery.getReferenceCount();

  	minhash_distance::CompareOutput * distances;
    double distanceMax = 1;
    double pValueMax = 1;
    distances = compare(new CompareInput(sketchRef, sketchQuery, 0, 0, 1, parametersNew, distanceMax, pValueMax));

    return distances;


  /********Write this output in the passes_threshold function ***************/


    // uint64_t i = distances->indexQuery;
    // uint64_t j = distances->indexRef;
    
    // for ( uint64_t k = 0; k < distances->pairCount && i < distances->sketchQuery.getReferenceCount(); k++ )
    // {
    //     const CompareOutput::PairOutput * pair = &distances->pairs[k];


        
    //     if ( table && j == 0 )
    //     {
    //         cout << output->sketchQuery.getReference(i).name;
    //     }
        
    //     if ( table )
    //     {
    //         cout << '\t';
    
    //         if ( pair->pass )
    //         {
    //             cout << pair->distance;
    //         }
    //     }
    //     else if ( pair->pass )
    //     {
    //         cout << output->sketchRef.getReference(j).name << '\t' << output->sketchQuery.getReference(i).name << '\t' << pair->distance << '\t' << pair->pValue << '\t' << pair->numer << '/' << pair->denom << endl;
    //     }
    
    //     j++;
        
    //     if ( j == output->sketchRef.getReferenceCount() )
    //     {
    //         if ( table )
    //         {
    //             cout << endl;
    //         }
            
    //         j = 0;
    //         i++;
    //     }
    // }
    
  }






}






