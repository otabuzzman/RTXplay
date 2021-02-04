#ifndef OPTICS_H
#define OPTICS_H

typedef union {

	struct {
		float3 albedo ;
	} diffuse ;

	struct {
		float3 albedo ;
		float  fuzz ;
	} reflect ;

	struct {
		float  index ;
	} refract ;

} Optics ;

#endif // OPTICS_H
