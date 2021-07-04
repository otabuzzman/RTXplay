#ifndef PADDLE_H
#define PADDLE_H

#include <vector_types.h>

class Paddle {
	public:
		void set( const float3& hand ) ;
		void start( const int x, const int y ) ;
		void track( const int x, const int y ) ;
		float3 hand() const ;

	private:
		// reference frame coordinate system
		const float3 u_ = { 1.f, 0.f, 0.f } ;
		const float3 v_ = { 0.f, 0.f, 1.f } ;
		const float3 w_ = { 0.f, 1.f, 0.f } ;
		// polar coords of Veye-Vpat position
		float lo_ ;
		float la_ ;
		// x/ y memory
		int x_ ;
		int y_ ;
} ;

#endif // PADDLE_H
