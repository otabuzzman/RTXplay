#ifndef UTIL_HOST_H
#define UTIL_HOST_H

#include <memory>
#include <iomanip>
#include <iostream>

using std::shared_ptr ;
using std::make_shared ;

namespace util {

static __forceinline__ __host__ void optxLogStderr( unsigned int level, const char* tag, const char* message, void* /*cbdata*/ ) {
	std::cerr
		<< "OptiX API message : "
		<< std::setw(  2 ) << level << " : "
		<< std::setw( 12 ) << tag   << " : "
		<< message << "\n" ;
}

}

#endif // UTIL_HOST_H
