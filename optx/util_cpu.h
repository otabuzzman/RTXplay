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

extern "C" void optxInitialize()                                        noexcept( false ) ;
extern "C" void optxBuildAccelerationStructure ( const Things& things ) noexcept( false ) ;
extern "C" void optxCreateModules()                                     noexcept( false ) ;
extern "C" void optxCreateProgramGroups()                               noexcept( false ) ;
extern "C" void optxLinkPipeline()                                      noexcept( false ) ;
extern "C" void optxBuildShaderBindingTable( const Things& things )     noexcept( false ) ;
extern "C" void optxLaunchPipeline()             ;
extern "C" void optxCleanup()                    ;

}

#endif // UTIL_CPU_H
