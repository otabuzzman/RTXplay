#ifndef RGB_H
#define RGB_H

#include "V.h"

#include <iostream>

void rgb( std::ostream &out, C color ) {
	out << static_cast<int>( 255*color.x() ) << ' ' << static_cast<int>( 255*color.y() ) << ' ' << static_cast<int>( 255*color.z() ) << '\n' ;
}

#endif
