#ifndef UTIL_CPU_H
#define UTIL_CPU_H

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>

#define CUDA_CHECK( api ) \
	do {                                                   \
		cudaError_t e = api ;                              \
		if ( e != cudaSuccess ) {                          \
			std::ostringstream comment ;                   \
			comment                                        \
				<< "CUDA error: " << #api << " : "         \
				<< cudaGetErrorString( e ) << "\n" ;       \
			throw std::runtime_error( comment.str() ) ;    \
		}                                                  \
	} while ( false )

#define OPTX_CHECK( api )                                  \
	do {                                                   \
		if ( api != OPTIX_SUCCESS ) {                      \
			std::ostringstream comment ;                   \
			comment << "OPTX error: " << #api << "\n" ;    \
			throw std::runtime_error( comment.str() ) ;    \
		}                                                  \
	} while ( false )

#define OPTX_CHECK_LOG( api )                              \
	do {                                                   \
		char   log[512] ;                                  \
		size_t sizeof_log = sizeof( log ) ;                \
		if ( api != OPTIX_SUCCESS ) {                      \
			std::ostringstream comment ;                   \
			comment << "OPTX error: " << #api << " : "     \
			<< log << "\n" ;                               \
			throw std::runtime_error( comment.str() ) ;    \
		}                                                  \
	} while ( false )

#define GLFW_CHECK( api )                                  \
	do {                                                   \
		api ;                                              \
		const char* m ;                                    \
		if ( glfwGetError( &m ) != GLFW_NO_ERROR ) {       \
			std::ostringstream comment ;                   \
			comment << "GLFW error: " << m << std::endl ;  \
			throw std::runtime_error( comment.str() ) ;    \
		}                                                  \
	} while ( false )

#define GL_CHECK( api )                                     \
	do {                                                    \
		api ;                                               \
		GLenum e ;                                          \
		if ( ( e = glGetError() ) != GL_NO_ERROR ) {        \
			std::ostringstream comment ;                    \
			comment << "GL error: " << #api << " :" ;       \
			do {                                                                                                    \
				switch ( e ) {                                                                                      \
					case GL_INVALID_ENUM:                  comment << " GL_INVALID_ENUM" ; break ;                  \
					case GL_INVALID_VALUE:                 comment << " GL_INVALID_VALUE" ; break ;                 \
					case GL_INVALID_OPERATION:             comment << " GL_INVALID_OPERATION" ; break ;             \
					case GL_INVALID_FRAMEBUFFER_OPERATION: comment << " GL_INVALID_FRAMEBUFFER_OPERATION" ; break ; \
					case GL_OUT_OF_MEMORY:                 comment << " GL_OUT_OF_MEMORY" ; break ;                 \
					default:                               comment << " " << e << " (undefined)" ; break ;          \
				}                                                                                                   \
			} while ( ( e = glGetError() ) != GL_NO_ERROR ) ;                                                       \
			comment << std::endl ;                                                                                  \
			throw std::runtime_error( comment.str() ) ;     \
		}                                                   \
	} while ( false )

namespace util {

inline float rnd()                                   { return static_cast<float>( rand() )/( static_cast<float>( RAND_MAX )+1.f ) ; }
inline float rnd( const float min, const float max ) { return min+rnd()*( max-min ) ; }

inline float deg( const float rad ) { return rad*180.f/kPi ; }
inline float rad( const float deg ) { return deg*kPi/180.f ; }

inline static void optxLogStderr( unsigned int level, const char* tag, const char* message, void* /*cbdata*/ ) {
	std::cerr
		<< "OptiX API message : "
		<< std::setw(  2 ) << level << " : "
		<< std::setw( 12 ) << tag   << " : "
		<< message << "\n" ;
}

}

#endif // UTIL_CPU_H
