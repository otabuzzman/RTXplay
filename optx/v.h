#ifndef V_H
#define V_H

#include <cmath>

namespace V {

__forceinline__ __host__ __device__ float3 operator +  ( const float a,   const float3& b ) { return make_float3( a+b.x, a+b.y, a+b.z ) ; }
__forceinline__ __host__ __device__ float3 operator +  ( const float3& a, const float b   ) { return make_float3( a.x+b, a.y+b, a.z+b ) ; }
__forceinline__ __host__ __device__ float3 operator +  ( const float3& a, const float3& b ) { return make_float3( a.x+b.x, a.y+b.y, a.z+b.z ) ; }
__forceinline__ __host__ __device__ void   operator += ( float3& a, const float3& b ) { a.x+=b.x ; a.y+=b.y ; a.z+=b.z ; }

__forceinline__ __host__ __device__ float3 operator -  ( const float a,   const float3& b ) { return make_float3( a-b.x, a-b.y, a-b.z ) ; }
__forceinline__ __host__ __device__ float3 operator -  ( const float3& a, const float b   ) { return make_float3( a.x-b, a.y-b, a.z-b ) ; }
__forceinline__ __host__ __device__ float3 operator -  ( const float3& a, const float3& b ) { return make_float3( a.x-b.x, a.y-b.y, a.z-b.z ) ; }
__forceinline__ __host__ __device__ void   operator -= ( float3& a, const float3& b ) { a.x-=b.x ; a.y-=b.y ; a.z-=b.z ; }

__forceinline__ __host__ __device__ float3 operator *  ( const float a,   const float3& b ) { return make_float3( a*b.x, a*b.y, a*b.z ) ; }
__forceinline__ __host__ __device__ float3 operator *  ( const float3& a, const float b   ) { return make_float3( a.x*b, a.y*b, a.z*b ) ; }
__forceinline__ __host__ __device__ float3 operator *  ( const float3& a, const float3& b ) { return make_float3( a.x*b.x, a.y*b.y, a.z*b.z ) ; }
__forceinline__ __host__ __device__ void   operator *= ( float3& a, const float3& b ) { a.x*=b.x ; a.y*=b.y ; a.z*=b.z ; }

__forceinline__ __host__ __device__ float3 operator /  ( const float a,   const float3& b ) { return make_float3( a/b.x, a/b.y, a/b.z ) ; }
__forceinline__ __host__ __device__ float3 operator /  ( const float3& a, const float b   ) { return 1.f/b*a ; }
__forceinline__ __host__ __device__ void   operator /= ( float3& a, const float b   ) { 1.f/b*a ; }
__forceinline__ __host__ __device__ float3 operator /  ( const float3& a, const float3& b ) { return make_float3( a.x/b.x, a.y/b.y, a.z/b.z ) ; }

__forceinline__ __host__ __device__ float  dot  ( const float3& a, const float3& b ) { return a.x*b.x+a.y*b.y+a.z*b.z ; }
__forceinline__ __host__ __device__ float  len  ( const float3& v )                  { return sqrtf( dot( v, v ) ) ; }
__forceinline__ __host__ __device__ float3 cross( const float3& a, const float3& b ) { return make_float3( a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x ) ; }
__forceinline__ __host__ __device__ float3 unitV( const float3& v )                  { return 1.f/len( v )*v ; }

}

#endif // V_H
