#include <iostream>

int main() {
	int w = 256, h = 256, c = 255 ;

	std::cout
		<< "P3\n"	// magic PPM header
		<< w		// width in pixels
		<< ' '
		<< h		// height in pixels
		<< '\n'
		<< c		// color maximum value
		<< '\n' ;

	for ( int j = h-1 ; j>=0 ; --j ) {
		for ( int i = 0 ; i<w ; ++i ) {
			auto r = (double) i/( w-1 ) ;
			auto g = (double) j/( h-1 ) ;
			auto b = .25 ;

			std::cout
				<< static_cast<int>( c*r )
				<< ' '
				<< static_cast<int>( c*g )
				<< ' '
				<< static_cast<int>( c*b )
				<< '\n' ;
		}
	}
}
