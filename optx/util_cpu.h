#ifndef UTIL_CPU_H
#define UTIL_CPU_H

#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace util {

inline float rnd()                                   { return static_cast<float>( rand() )/( static_cast<float>( RAND_MAX )+1.f ) ; }
inline float rnd( const float min, const float max ) { return min+rnd()*( max-min ) ; }

inline static void optxLogStderr( unsigned int level, const char* tag, const char* message, void* /*cbdata*/ ) {
	std::cerr
		<< "OptiX API message : "
		<< std::setw(  2 ) << level << " : "
		<< std::setw( 12 ) << tag   << " : "
		<< message << "\n" ;
}

}

#endif // UTIL_CPU_H
