#ifndef V_H
#define V_H

// system includes
// none

// subsystem includes
// CUDA
#include <vector_functions.h>
#ifdef CURAND
#include <curand_kernel.h>
#endif // CURAND
#include <vector_types.h>

// local includes
#ifndef CURAND
#include "frand48.h"
#endif // CURAND
#include "util.h"

// file specific includes
// none

namespace V {

__forceinline__ __host__ __device__ float3 operator +  ( const float a,   const float3& b ) { return make_float3( a+b.x, a+b.y, a+b.z ) ; }
__forceinline__ __host__ __device__ float3 operator +  ( const float3& a, const float b   ) { return make_float3( a.x+b, a.y+b, a.z+b ) ; }
__forceinline__ __host__ __device__ float3 operator +  ( const float3& a, const float3& b ) { return make_float3( a.x+b.x, a.y+b.y, a.z+b.z ) ; }
__forceinline__ __host__ __device__ void   operator += ( float3& a, const float3& b )       { a.x+=b.x ; a.y+=b.y ; a.z+=b.z ; }

__forceinline__ __host__ __device__ float3 operator -  ( const float3& a )                  { return make_float3( -a.x, -a.y, -a.z ) ; ; }

__forceinline__ __host__ __device__ float3 operator -  ( const float a,   const float3& b ) { return make_float3( a-b.x, a-b.y, a-b.z ) ; }
__forceinline__ __host__ __device__ float3 operator -  ( const float3& a, const float b   ) { return make_float3( a.x-b, a.y-b, a.z-b ) ; }
__forceinline__ __host__ __device__ float3 operator -  ( const float3& a, const float3& b ) { return make_float3( a.x-b.x, a.y-b.y, a.z-b.z ) ; }
__forceinline__ __host__ __device__ void   operator -= ( float3& a, const float3& b )       { a.x-=b.x ; a.y-=b.y ; a.z-=b.z ; }

__forceinline__ __host__ __device__ float3 operator *  ( const float a,   const float3& b ) { return make_float3( a*b.x, a*b.y, a*b.z ) ; }
__forceinline__ __host__ __device__ float3 operator *  ( const float3& a, const float b   ) { return make_float3( a.x*b, a.y*b, a.z*b ) ; }
__forceinline__ __host__ __device__ float3 operator *  ( const float3& a, const float3& b ) { return make_float3( a.x*b.x, a.y*b.y, a.z*b.z ) ; }
__forceinline__ __host__ __device__ void   operator *= ( float3& a, const float3& b )       { a.x*=b.x ; a.y*=b.y ; a.z*=b.z ; }

__forceinline__ __host__ __device__ float3 operator /  ( const float a,   const float3& b ) { return make_float3( a/b.x, a/b.y, a/b.z ) ; }
__forceinline__ __host__ __device__ float3 operator /  ( const float3& a, const float b   ) { return 1.f/b*a ; }
__forceinline__ __host__ __device__ void   operator /= ( float3& a, const float b   )       { 1.f/b*a ; }
__forceinline__ __host__ __device__ float3 operator /  ( const float3& a, const float3& b ) { return make_float3( a.x/b.x, a.y/b.y, a.z/b.z ) ; }

__forceinline__ __host__ __device__ float  dot  ( const float3& a, const float3& b ) { return a.x*b.x+a.y*b.y+a.z*b.z ; }
__forceinline__ __host__ __device__ float  len  ( const float3& v )                  { return sqrtf( dot( v, v ) ) ; }
__forceinline__ __host__ __device__ float3 cross( const float3& a, const float3& b ) { return make_float3( a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x ) ; }
__forceinline__ __host__ __device__ float3 unitV( const float3& v )                  { return 1.f/len( v )*v ; }

__forceinline__ __host__ __device__ bool   near0( const float3& v )                  { return ( fabsf( v.x )<util::kNear0 ) && ( fabsf( v.y )<util::kNear0 ) && ( fabsf( v.z )<util::kNear0 ) ; }

#ifdef __CUDACC__

#ifdef CURAND
__forceinline__ __device__ float3 rnd( curandState *state )                                   { return make_float3( util::rnd( state ), util::rnd( state ), util::rnd( state ) ) ; }
__forceinline__ __device__ float3 rnd( const float min, const float max, curandState *state ) { return make_float3( util::rnd( min, max, state ), util::rnd( min, max, state ), util::rnd( min, max, state ) ) ; }

// random V (float3) in unit sphere
__forceinline__ __device__ float3 rndVin1sphere( curandState *state ) { while ( true ) { auto v = rnd( -1.f, 1.f, state ) ; if ( 1.f>dot( v, v ) ) return v ; } }
// random V (float3) on unit sphere (chapter 8.5)
__forceinline__ __device__ float3 rndVon1sphere( curandState *state ) { return unitV( rndVin1sphere( state ) ) ; }

// random V (float3) in unit disk (chapter 12.2)
__forceinline__ __device__ float3 rndVin1disk( curandState *state )   { while ( true ) { auto v = make_float3( util::rnd( -1.f, 1.f, state ), util::rnd( -1.f, 1.f, state ), 0.f ) ; if ( 1.f>dot( v, v ) ) return v ; } }
#else
__forceinline__ __device__ float3 rnd( Frand48 *state )                                       { return make_float3( util::rnd( state ), util::rnd( state ), util::rnd( state ) ) ; }
__forceinline__ __device__ float3 rnd( const float min, const float max, Frand48 *state )     { return make_float3( util::rnd( min, max, state ), util::rnd( min, max, state ), util::rnd( min, max, state ) ) ; }

__forceinline__ __device__ float3 rndVin1sphere( Frand48 *state )     { while ( true ) { auto v = rnd( -1.f, 1.f, state ) ; if ( 1.f>dot( v, v ) ) return v ; } }
__forceinline__ __device__ float3 rndVon1sphere( Frand48 *state )     { return unitV( rndVin1sphere( state ) ) ; }

__forceinline__ __device__ float3 rndVin1disk( Frand48 *state )       { while ( true ) { auto v = make_float3( util::rnd( -1.f, 1.f, state ), util::rnd( -1.f, 1.f, state ), 0.f ) ; if ( 1.f>dot( v, v ) ) return v ; } }
#endif // CURAND

#else

inline float3 rnd()                                   { return make_float3( util::rnd(), util::rnd(), util::rnd() ) ; }
inline float3 rnd( const float min, const float max ) { return make_float3( util::rnd( min, max ), util::rnd( min, max ), util::rnd( min, max ) ) ; }

#endif // __CUDACC__

}

#endif // V_H
