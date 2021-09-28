#ifndef OPTICS_H
#define OPTICS_H

struct Diffuse {
	float3 albedo ; // wavefront .mtl : Kd
} ;

struct Reflect {
	float3 albedo ; // wavefront .mtl : Kd
	float  fuzz ;   // wavefront .mtl : sharpness
} ;

struct Refract {
	float  index ;  // wavefront .mtl : Ni
} ;

// Optics data passed for each thing via SBT record to closest hit program
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

#endif // OPTICS_H
