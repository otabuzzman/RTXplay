#ifndef UTIL_CPU_H
#define UTIL_CPU_H

#include <iomanip>
#include <iostream>

#include "things.h"

namespace util {

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
