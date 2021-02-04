#ifndef UTIL_CPU_H
#define UTIL_CPU_H

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "things.h"

namespace util {

inline float rnd()                                   { return static_cast<float>( rand() )/( static_cast<float>( RAND_MAX )+1.f ) ; }
inline float rnd( const float min, const float max ) { return min+rnd()*( max-min ) ; }

static inline void optxLogStderr( unsigned int level, const char* tag, const char* message, void* /*cbdata*/ ) {
	std::cerr
		<< "OptiX API message : "
		<< std::setw(  2 ) << level << " : "
		<< std::setw( 12 ) << tag   << " : "
		<< message << "\n" ;
}

void optxInitialize()                                                    noexcept( false ) ;
void optxBuildAccelerationStructure ( const Things& things )             noexcept( false ) ;
void optxCreateModules()                                                 noexcept( false ) ;
void optxCreateProgramGroups()                                           noexcept( false ) ;
void optxLinkPipeline()                                                  noexcept( false ) ;
void optxBuildShaderBindingTable( const Things& things )                 noexcept( false ) ;
const std::vector<uchar4> optxLaunchPipeline( const int w, const int h )                   ;
void optxCleanup()                                                       noexcept( false ) ;

}

#endif // UTIL_CPU_H
