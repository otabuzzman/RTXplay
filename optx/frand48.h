#ifndef FRAND48_H
#define FRAND48_H

class Frand48 {
	public:
		__forceinline__ __device__ void init( const unsigned int seed = 4711 ) {
			state = seed ;

			// http://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf
			for ( int r = 0 ; 16>r ; r++ )
				( *this )() ;
		}

		__forceinline__ __device__ float operator () () {
			state = 0x5DEECE66Dull*state+0xBull ;

			return float( state&0xFFFFFFFFFFFFull )/float( 0xFFFFFFFFFFFFull+1ull ) ;
		}

	private:
		unsigned long long state ;
} ;

#endif // FRAND48_H
