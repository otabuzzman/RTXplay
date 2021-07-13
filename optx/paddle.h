#ifndef PADDLE_H
#define PADDLE_H

#include <vector_types.h>

class Paddle {
	public:
		Paddle( const float3& eye, const float3& pat, const float3& vup ) ;

		void start( const int x, const int y ) ;
		float3 move( const int x, const int y ) ;
		float3 roll( const int s ) ;

	private:
		// reference frame coordinate system
		const float3 u_ = { 1.f, 0.f, 0.f } ;
		const float3 v_ = { 0.f, 0.f, 1.f } ;
		const float3 w_ = { 0.f, 1.f, 0.f } ;
		// move memory
		float lo_ ;
		float la_ ;
		int x_ ;
		int y_ ;
		// roll memory
		float3 vup_ ;
		float  phi_ ;
} ;

#endif // PADDLE_H
