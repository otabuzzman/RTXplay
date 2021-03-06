#ifndef RTWO_H
#define RTWO_H

struct LpGeneral { // launch parameter
	uchar4*                image ;
	unsigned int           image_w ;
	unsigned int           image_h ;

	Camera                 camera ;

	unsigned int           spp ;   // samples per pixel
	unsigned int           depth ; // recursion depth

	OptixTraversableHandle as_handle ;
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

#endif // RTWO_H
