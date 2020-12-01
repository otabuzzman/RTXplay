#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <limits>
#include <memory>

using std::shared_ptr ;
using std::make_shared ;
using std::sqrt ;

const double INF = std::numeric_limits<double>::infinity() ;
const double PI = 3.141592653589793238 ;

inline double deg2rad( double d ) {
    return d * PI/180. ;
}

#include "Ray.h"
#include "V.h"

#endif