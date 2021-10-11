#ifndef THING_H
#define THING_H

// system includes
#include <vector>

// subsystem includes
// CUDA
#include <vector_types.h>

// local includes
#include "util.h"

// file specific includes
// none

struct Diffuse {
	float3 albedo ; // wavefront MTL : Kd
} ;

struct Reflect {
	float3 albedo ; // wavefront MTL : Kd
	float  fuzz ;   // wavefront MTL : sharpness
} ;

struct Refract {
	float  index ;  // wavefront MTL : Ni
} ;

struct Optics {
	enum {
		TYPE_DIFFUSE,
		TYPE_REFLECT,
		TYPE_REFRACT,
		TYPE_NUM
	} ;

	int type ;

	union {
		Diffuse diffuse ;
		Reflect reflect ;
		Refract refract ;
	} ;
} ;

struct Thing {
	float3* vces = nullptr ;
	uint3*  ices = nullptr ;

	Optics  optics ;

	float  transform[12] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0
	} ;
} ;

#endif // THING_H
