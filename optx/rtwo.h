#ifndef RTWO_H
#define RTWO_H

// system includes
// none

// subsystem includes
// CUDA
#include <vector_types.h>
// OptiX
#include <optix.h>

// local includes
#include "camera.h"
#include "frand48.h"
#include "thing.h"

// file specific includes
// none

struct LpGeneral { // launch parameter
	uchar4*                image ;
	unsigned int           image_w ;
	unsigned int           image_h ;

	float3*                rawRGB ; // from RG prgram

	// denoiser guide layers (average values if SPP>1)
	float3*                normals ; // pixel normals
	float3*                albedos ; // pixel color approximations

	Camera                 camera ;

	unsigned int           spp ;   // samples per pixel
	unsigned int           depth ; // recursion depth

	OptixTraversableHandle as_handle ;

	// arbitrary output variables (AOV)
	unsigned int*          rpp ;   // rays per pixel
} ;

template <typename T>
struct SbtRecord {
	__align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE] ;

	T data ;
} ;

struct NoData {} ;

typedef SbtRecord<NoData> SbtRecordRG ; // Ray Generation program group SBT record type
typedef SbtRecord<float3> SbtRecordMS ; // Miss program group SBT record type
typedef SbtRecord<Thing>  SbtRecordHG ; // Hit Group program group SBT record type

// used by iterative programs (instead of separate ray payloads)
struct RayParam {
	float3       color ;
	float3       hit ;
	float3       dir ;
#ifdef CURAND
	curandState  rng ;
#else
	Frand48      rng ;
#endif // CURAND

#define RP_STAT_NONE 0 // default
#define RP_STAT_CONT 1 // HG shader returned color
#define RP_STAT_STOP 2 // HG shader reached end of trace
#define RP_STAT_MISS 4 // MS shader returned color (end of trace)
	unsigned int stat = RP_STAT_NONE ;

	// denoiser guide values
	float3       normal ;
	float3       albedo ;
} ;

// used by recursive programs (as additional separate ray payload)
struct DenoiserGuideValues {
	float3 normal ;
	float3 albedo ;
} ;

#endif // RTWO_H
