#ifndef HOIST_H
#define HOIST_H

// system includes
// none

// subsystem includes
// none

// local includes
#include "thing.h"

// file specific includes
// none

struct Hoist : public Thing {
	unsigned int num_vces = 0 ;
	unsigned int num_ices = 0 ;
} ;

#endif // HOIST_H
