#ifndef RGB_H
#define RGB_H

#include "util.h"

#include <iostream>

void rgb( std::ostream &out, C color ) {
	out << static_cast<int>( 255*color.x() ) << ' '
		<< static_cast<int>( 255*color.y() ) << ' '
		<< static_cast<int>( 255*color.z() ) << '\n' ;
}

void rgb( std::ostream &out, C color, int spp ) {
	auto r = color.x() ; auto g = color.y() ; auto b = color.z() ;

	auto s = 1./spp ;
	r *= s ; g *= s ; b *= s ;

	out << static_cast<int>( 256 * clamp( r, 0, .999 ) ) << ' '
		<< static_cast<int>( 256 * clamp( g, 0, .999 ) ) << ' '
		<< static_cast<int>( 256 * clamp( b, 0, .999 ) ) << '\n' ;
}

#endif
