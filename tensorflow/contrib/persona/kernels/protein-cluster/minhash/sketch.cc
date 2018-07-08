
#include "Sketch.h"
#include <unistd.h>
#include <zlib.h>
#include <stdio.h>
#include <iostream>
#include <fcntl.h>
#include <map>
#include "kseq.h"
#include "MurmurHash3.h"
#include <assert.h>
#include <queue>
#include <deque>
#include <set>
#include <math.h>

using namespace std;


void reverseComplement(const char * src, char * dest, int length)
{
    for ( int i = 0; i < length; i++ )
    {
        char base = src[i];
        
        switch ( base )
        {
            case 'A': base = 'T'; break;
            case 'C': base = 'G'; break;
            case 'G': base = 'C'; break;
            case 'T': base = 'A'; break;
            default: break;
        }
        
        dest[length - i - 1] = base;
    }
}


void addMinHashes(MinHashHeap & minHashHeap, char * seq, uint64_t length, const Sketch::Parameters & parameters)
{
    int kmerSize = parameters.kmerSize;
    uint64_t mins = parameters.minHashesPerWindow;
    bool noncanonical = parameters.noncanonical;
    
    // Determine the 'mins' smallest hashes, including those already provided
    // (potentially replacing them). This allows min-hash sets across multiple
    // sequences to be determined.
    
    // uppercase TODO: alphabets?
    //
    for ( uint64_t i = 0; i < length; i++ )
    {
        if ( ! parameters.preserveCase && seq[i] > 96 && seq[i] < 123 )
        {
            seq[i] -= 32;
        }
    }
    
    char * seqRev;
    
    if ( ! noncanonical )
    {
    	seqRev = new char[length];
        reverseComplement(seq, seqRev, length);
    }
    
    for ( uint64_t i = 0; i < length - kmerSize + 1; i++ )
    {
        bool useRevComp = false;
        bool debug = false;
        
		// repeatedly skip kmers with bad characters
		
		bool bad = false;
		
		for ( uint64_t j = i; j < i + kmerSize && i + kmerSize <= length; j++ )
		{
			if ( ! parameters.alphabet[seq[j]] )
			{
				i = j; // skip to past the bad character
				bad = true;
				break;
			}
		}
		
		if ( bad )
		{
			continue;
		}
	
		if ( i + kmerSize > length )
		{
			// skipped to end
			break;
		}
            
        if ( ! noncanonical )
        {
            useRevComp = true;
            bool prefixEqual = true;
        
            if ( debug ) {for ( uint64_t j = i; j < i + kmerSize; j++ ) { cout << *(seq + j); } cout << endl;}
        
            for ( uint64_t j = 0; j < kmerSize; j++ )
            {
                char base = seq[i + j];
                char baseMinus = seqRev[length - i - kmerSize + j];
            
                if ( debug ) cout << baseMinus;
            
                if ( prefixEqual && baseMinus > base )
                {
                    useRevComp = false;
                    break;
                }
            
                if ( prefixEqual && baseMinus < base )
                {
                    prefixEqual = false;
                }
            }
        
            if ( debug ) cout << endl;
        }
        
        const char * kmer = useRevComp ? seqRev + length - i - kmerSize : seq + i;
        bool filter = false;
        
        hash_u hash = getHash(useRevComp ? seqRev + length - i - kmerSize : seq + i, kmerSize, parameters.seed, parameters.use64);
        
        if ( debug ) cout << endl;
        
		minHashHeap.tryInsert(hash);
    }
    
    if ( ! noncanonical )
    {
        delete [] seqRev;
    }
}


void addMinHashes(MinHashHeap & minHashHeap, char * seq, uint64_t length, const Sketch::Parameters & parameters)
{
    int kmerSize = parameters.kmerSize;
    uint64_t mins = parameters.minHashesPerWindow;
    bool noncanonical = parameters.noncanonical;
    
    // Determine the 'mins' smallest hashes, including those already provided
    // (potentially replacing them). This allows min-hash sets across multiple
    // sequences to be determined.
    
    // uppercase TODO: alphabets?
    //
    for ( uint64_t i = 0; i < length; i++ )
    {
        if ( ! parameters.preserveCase && seq[i] > 96 && seq[i] < 123 )
        {
            seq[i] -= 32;
        }
    }
    
    char * seqRev;
    
    if ( ! noncanonical )
    {
    	seqRev = new char[length];
        reverseComplement(seq, seqRev, length);
    }
    
    for ( uint64_t i = 0; i < length - kmerSize + 1; i++ )
    {
        bool useRevComp = false;
        bool debug = false;
        
		// repeatedly skip kmers with bad characters
		
		bool bad = false;
		
		for ( uint64_t j = i; j < i + kmerSize && i + kmerSize <= length; j++ )
		{
			if ( ! parameters.alphabet[seq[j]] )
			{
				i = j; // skip to past the bad character
				bad = true;
				break;
			}
		}
		
		if ( bad )
		{
			continue;
		}
	
		if ( i + kmerSize > length )
		{
			// skipped to end
			break;
		}
            
        if ( ! noncanonical )
        {
            useRevComp = true;
            bool prefixEqual = true;
        
            if ( debug ) {for ( uint64_t j = i; j < i + kmerSize; j++ ) { cout << *(seq + j); } cout << endl;}
        
            for ( uint64_t j = 0; j < kmerSize; j++ )
            {
                char base = seq[i + j];
                char baseMinus = seqRev[length - i - kmerSize + j];
            
                if ( debug ) cout << baseMinus;
            
                if ( prefixEqual && baseMinus > base )
                {
                    useRevComp = false;
                    break;
                }
            
                if ( prefixEqual && baseMinus < base )
                {
                    prefixEqual = false;
                }
            }
        
            if ( debug ) cout << endl;
        }
        
        const char * kmer = useRevComp ? seqRev + length - i - kmerSize : seq + i;
        bool filter = false;
        
        hash_u hash = getHash(useRevComp ? seqRev + length - i - kmerSize : seq + i, kmerSize, parameters.seed, parameters.use64);
        
        if ( debug ) cout << endl;
        
		minHashHeap.tryInsert(hash);
    }
    
    if ( ! noncanonical )
    {
        delete [] seqRev;
    }
}



void setMinHashesForReference(Sketch::Reference & reference, const MinHashHeap & hashes)
{
    HashList & hashList = reference.hashesSorted;
    hashList.clear();
    hashes.toHashList(hashList);
    hashes.toCounts(reference.counts);
    hashList.sort();
}



void getMinHashPositions(vector<Sketch::PositionHash> & positionHashes, char * seq, uint32_t length, const Sketch::Parameters & parameters, int verbosity)
{
    // Find positions whose hashes are min-hashes in any window of a sequence
    
    int kmerSize = parameters.kmerSize;
    int mins = parameters.minHashesPerWindow;
    int windowSize = parameters.windowSize;
    
    int nextValidKmer = 0;
    
    if ( windowSize > length - kmerSize + 1 )
    {
        windowSize = length - kmerSize + 1;
    }
    
    if ( verbosity > 1 ) cout << seq << endl << endl;
    
    // Associate positions with flags so they can be marked as min-hashes
    // at any point while the window is moved across them
    //
    struct CandidateLocus
    {
        CandidateLocus(int positionNew)
            :
            position(positionNew),
            isMinmer(false)
            {}
        
        int position;
        bool isMinmer;
    };
    
    // All potential min-hash loci in the current window organized by their
    // hashes so repeats can be grouped and so the sorted keys can be used to
    // keep track of the current h bottom hashes. A deque is used here (rather
    // than a standard queue) for each list of candidate loci for the sake of
    // debug output; the performance difference seems to be negligible.
    //
    map<Sketch::hash_t, deque<CandidateLocus>> candidatesByHash;
    
    // Keep references to the candidate loci in the map in the order of the
    // current window, allowing them to be popped off in the correct order as
    // the window is incremented.
    //
    queue<map<Sketch::hash_t, deque<CandidateLocus>>::iterator> windowQueue;
    
    // Keep a reference to the "hth" min-hash to determine efficiently whether
    // new hashes are min-hashes. It must be decremented when a hash is inserted
    // before it.
    //
    map<Sketch::hash_t, deque<CandidateLocus>>::iterator maxMinmer = candidatesByHash.end();
    
    // A reference to the list of potential candidates that has just been pushed
    // onto the back of the rolling window. During the loop, it will be assigned
    // to either an existing list (if the kmer is repeated), a new list, or a
    // dummy iterator (for invalid kmers).
    //
    map<Sketch::hash_t, deque<CandidateLocus>>::iterator newCandidates;
    
    int unique = 0;
    
    for ( int i = 0; i < length - kmerSize + 1; i++ )
    {
        // Increment the next valid kmer if needed. Invalid kmers must still be
        // processed to keep the queue filled, but will be associated with a
        // dummy iterator. (Currently disabled to allow all kmers; see below)
        //
        if ( i >= nextValidKmer )
        {
            for ( int j = i; j < i + kmerSize; j++ )
            {
                char c = seq[j];
                
                if ( c != 'A' && c != 'C' && c != 'G' && c != 'T' )
                {
                    // Uncomment to skip invalid kmers
                    //
                    //nextValidKmer = j + 1;
                    
                    break;
                }
            }
        }
        
        if ( i < nextValidKmer && verbosity > 1 )
        {
            cout << "  [";
        
            for ( int j = i; j < i + kmerSize; j++ )
            {
                cout << seq[j];
            }
            
            cout << "]" << endl;
        }
        
        if ( i >= nextValidKmer )
        {
            Sketch::hash_t hash = getHash(seq + i, kmerSize, parameters.seed, parameters.use64).hash64; // TODO: dynamic
            
            if ( verbosity > 1 )
            {
                cout << "   ";
            
                for ( int j = i; j < i + kmerSize; j++ )
                {
                    cout << seq[j]; 
                }
            
                cout << "   " << i << '\t' << hash << endl;
            }
            
            // Get the list of candidate loci for the current hash (if it is a
            // repeat) or insert a new list.
            //
            pair<map<Sketch::hash_t, deque<CandidateLocus>>::iterator, bool> inserted =
                candidatesByHash.insert(pair<Sketch::hash_t, deque<CandidateLocus>>(hash, deque<CandidateLocus>()));
            newCandidates = inserted.first;
            
            // Add the new candidate locus to the list
            //
            newCandidates->second.push_back(CandidateLocus(i));
            
            if
            (
                inserted.second && // inserted; decrement maxMinmer if...
                (
                    (
                        // ...just reached number of mins
                        
                        maxMinmer == candidatesByHash.end() &&
                        candidatesByHash.size() == mins
                    ) ||
                    (
                        // ...inserted before maxMinmer
                        
                        maxMinmer != candidatesByHash.end() &&
                        newCandidates->first < maxMinmer->first
                    )
                )
            )
            {
                maxMinmer--;
                
                if ( i >= windowSize )
                {
                    unique++;
                }
            }
        }
        else
        {
            // Invalid kmer; use a dummy iterator to pad the queue
            
            newCandidates = candidatesByHash.end();
        }
        
        // Push the new reference to the list of candidate loci for the new kmer
        // on the back of the window to roll it.
        //
        windowQueue.push(newCandidates);
        
        // A reference to the front of the window, to be popped off if the
        // window has grown to full size. This reference can be a dummy if the
        // window is not full size or if the front of the window is a dummy
        // representing an invalid kmer.
        //
        map<Sketch::hash_t, deque<CandidateLocus>>::iterator windowFront = candidatesByHash.end();
        
        if ( windowQueue.size() > windowSize )
        {
            windowFront = windowQueue.front();
            windowQueue.pop();
            
            if ( verbosity > 1 ) cout << "   \tPOP: " << windowFront->first << endl;
        }
        
        if ( windowFront != candidatesByHash.end() )
        {
            deque<CandidateLocus> & frontCandidates = windowFront->second;
            
            if ( frontCandidates.front().isMinmer )
            {
                if ( verbosity > 1 ) cout << "   \t   minmer: " << frontCandidates.front().position << '\t' << windowFront->first << endl;
                positionHashes.push_back(Sketch::PositionHash(frontCandidates.front().position, windowFront->first));
            }
            
            if ( frontCandidates.size() > 1 )
            {
                frontCandidates.pop_front();
                
                // Since this is a repeated hash, only the locus in the front of
                // the list was considered min-hash loci. Check if the new front
                // will become a min-hash so it can be flagged.
                //
                if ( maxMinmer == candidatesByHash.end() || ( i >= windowSize && windowFront->first <= maxMinmer->first) )
                {
                    frontCandidates.front().isMinmer = true;
                }
            }
            else
            {
                // The list for this hash is no longer needed; destroy it,
                // repositioning the reference to the hth min-hash if
                // necessary.
                
                if ( maxMinmer != candidatesByHash.end() && windowFront->first <= maxMinmer->first )
                {
                    maxMinmer++;
                    
                    if ( maxMinmer != candidatesByHash.end() )
                    {
                        maxMinmer->second.front().isMinmer = true;
                    }
                    
                    unique++;
                }
            
                candidatesByHash.erase(windowFront);
            }
        }
        
        if ( i == windowSize - 1 )
        {
            // first complete window; mark min-hashes
            
            for ( map<Sketch::hash_t, deque<CandidateLocus>>::iterator j = candidatesByHash.begin(); j != maxMinmer; j++ )
            {
                j->second.front().isMinmer = true;
            }
            
            if ( maxMinmer != candidatesByHash.end() )
            {
                maxMinmer->second.front().isMinmer = true;
            }
            
            unique++;
        }
        
        // Mark the candidate that was pushed on the back of the window queue
        // earlier as a min-hash if necessary
        //
        if ( newCandidates != candidatesByHash.end() && i >= windowSize && (maxMinmer == candidatesByHash.end() || newCandidates->first <= maxMinmer->first) )
        {
            newCandidates->second.front().isMinmer = true;
        }
        
        if ( verbosity > 1 )
        {
            for ( map<Sketch::hash_t, deque<CandidateLocus>>::iterator j = candidatesByHash.begin(); j != candidatesByHash.end(); j++ )
            {
                cout << "   \t" << j->first;
                
                if ( j == maxMinmer )
                {
                     cout << "*";
                }
                
                for ( deque<CandidateLocus>::iterator k = j->second.begin(); k != j->second.end(); k++ )
                {
                    cout << '\t' << k->position;
                    
                    if ( k->isMinmer )
                    {
                        cout << '!';
                    }
                }
                
                cout << endl;
            }
        }
    }
    
    // finalize remaining min-hashes from the last window
    //
    while ( windowQueue.size() > 0 )
    {
        map<Sketch::hash_t, deque<CandidateLocus>>::iterator windowFront = windowQueue.front();
        windowQueue.pop();
        
        if ( windowFront != candidatesByHash.end() )
        {
            deque<CandidateLocus> & frontCandidates = windowFront->second;
            
            if ( frontCandidates.size() > 0 )
            {
                if ( frontCandidates.front().isMinmer )
                {
                    if ( verbosity > 1 ) cout << "   \t   minmer:" << frontCandidates.front().position << '\t' << windowFront->first << endl;
                    positionHashes.push_back(Sketch::PositionHash(frontCandidates.front().position, windowFront->first));
                }
                
                frontCandidates.pop_front();
            }
        }
    }
    
    if ( verbosity > 1 )
    {
        cout << endl << "Minmers:" << endl;
    
        for ( int i = 0; i < positionHashes.size(); i++ )
        {
            cout << "   " << positionHashes.at(i).position << '\t' << positionHashes.at(i).hash << endl;
        }
        
        cout << endl;
    }
    
    if ( verbosity > 0 ) cout << "   " << positionHashes.size() << " minmers across " << length - windowSize - kmerSize + 2 << " windows (" << unique << " windows with distinct minmer sets)." << endl << endl;
}




int Sketch::init(std::string fileNameNew, char * seqNew, uint64_t lengthNew, const std::string & nameNew, const std::string & commentNew, const Sketch::Parameters & parametersNew)
{

	//check the exact arguements requied by sketchInput here and send those arguments to sketch::Inti from minhash_distance.cpp
	parameters = parametersNew;
	Sketch::SketchOutput * outputstructure= sketchSequence(new SketchInput("", seqNew, lengthNew, "", "", parametersNew));
	useThreadOutput(outputstructure);
}


void Sketch::useThreadOutput(SketchOutput * output)
{
	references.insert(references.end(), output->references.begin(), output->references.end());
	positionHashesByReference.insert(positionHashesByReference.end(), output->positionHashesByReference.begin(), output->positionHashesByReference.end());
	delete output;
}

Sketch::SketchOutput * sketchSequence(Sketch::SketchInput * input)
{
	const Sketch::Parameters & parameters = input->parameters;
	
	Sketch::SketchOutput * output = new Sketch::SketchOutput();
	
	output->references.resize(1);
	Sketch::Reference & reference = output->references[0];
	
	reference.length = input->length;
	reference.name = input->name;
	reference.comment = input->comment;
	reference.hashesSorted.setUse64(parameters.use64);
	
	if ( parameters.windowed )
	{
		output->positionHashesByReference.resize(1);
		getMinHashPositions(output->positionHashesByReference[0], input->seq, input->length, parameters, 0);
	}
	else
	{
	    MinHashHeap minHashHeap(parameters.use64, parameters.minHashesPerWindow, parameters.reads ? parameters.minCov : 1);
        addMinHashes(minHashHeap, input->seq, input->length, parameters);
		setMinHashesForReference(reference, minHashHeap);
	}
	
	return output;
}
