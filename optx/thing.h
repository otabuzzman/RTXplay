#ifndef THING_H
#define THING_H

// system includes
// none

// subsystem includes
// CUDA
#include <vector_types.h>

// local includes
// none

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
	struct { // std::vector<float3> device equivalent
		float3*      data ;
		unsigned int size ;
	} vces ;
	struct { // std::vector<uint3> device equivalent
		uint3*       data ;
		unsigned int size ;
	} ices ;

	Optics optics ;

	float  transform[12] = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0
	} ;
} ;

#endif // THING_H
