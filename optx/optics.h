#ifndef OPTICS_H
#define OPTICS_H

enum {
	OPTICS_TYPE_DIFFUSE,
	OPTICS_TYPE_REFLECT,
	OPTICS_TYPE_REFRACT,
	OPTICS_TYPE_NUM
} ;

typedef struct {
	int type ;

	union {
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
	} ;
} Optics ;

#endif // OPTICS_H
