#ifndef UTIL_CPU_H
#define UTIL_CPU_H

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

#define CUDA_CHECK( api ) \
			if ( true ) {                                       \
				cudaError_t e = api ;                           \
				if ( e != cudaSuccess ) {                       \
					std::ostringstream comment ;                \
					comment                                     \
						<< "CUDA error: " << #api << " : "      \
						<< cudaGetErrorString( e ) << "\n" ;    \
					throw std::runtime_error( comment.str() ) ; \
				}                                               \
			} else

#define OPTX_CHECK( api )                                       \
			if ( true ) {                                       \
				if ( api != OPTIX_SUCCESS ) {                   \
					std::ostringstream comment ;                \
					comment << "OPTX error: " << #api << "\n" ; \
					throw std::runtime_error( comment.str() ) ; \
				}                                               \
			} else

#define OPTX_CHECK_LOG( api )                                   \
			if ( true ) {                                       \
				char   log[512] ;                               \
				size_t sizeof_log = sizeof( log ) ;             \
				if ( api != OPTIX_SUCCESS ) {                   \
					std::ostringstream comment ;                \
					comment << "OPTX error: " << #api << " : "  \
					<< log << "\n" ;                            \
					throw std::runtime_error( comment.str() ) ; \
				}                                               \
			} else

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
