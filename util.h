#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

using std::shared_ptr ;
using std::make_shared ;

const double kInfinty = std::numeric_limits<double>::infinity() ;
const double kPi      = 3.141592653589793238 ;

inline double deg2rad( double d ) { return d*kPi/180. ; }

inline double rnd() { return rand()/( RAND_MAX+1. ) ; }
inline double rnd( double min, double max ) { return min+rnd()*( max-min ) ; }

inline double clamp( double x, double min, double max ) { return min>x ? min : x>max ? max : x ; }

#endif
