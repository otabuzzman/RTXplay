#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <cstdlib>
#include <limits>

const float kInfinty = std::numeric_limits<float>::infinity() ;
const float kPi      = 3.14159265358f ;

inline float deg2rad( float d ) { return d*kPi/180.f ; }

inline float rnd() { return rand()/( RAND_MAX+1.f ) ; }
inline float rnd( float min, float max ) { return min+rnd()*( max-min ) ; }

inline float clamp( float x, float min, float max ) { return min>x ? min : x>max ? max : x ; }

#endif
