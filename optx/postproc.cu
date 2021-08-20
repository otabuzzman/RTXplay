// sRGB conversion according to https://en.wikipedia.org/wiki/SRGB
static __forceinline__ __device__ void sRGB( const float3& rgb, float3& srgb ) {
	srgb.x = rgb.x<.0031308f ? 12.92f*rgb.x : 1.055f*powf( rgb.x, 1.f/2.4f )-.055f ;
	srgb.y = rgb.y<.0031308f ? 12.92f*rgb.y : 1.055f*powf( rgb.y, 1.f/2.4f )-.055f ;
	srgb.z = rgb.z<.0031308f ? 12.92f*rgb.z : 1.055f*powf( rgb.z, 1.f/2.4f )-.055f ;
}

static __forceinline__ __device__ void sRGB( const float3& rgb, uchar4& srgb ) {
	float3 c ;

	sRGB( rgb, c ) ;
	srgb.x = static_cast<unsigned char>( c.x*255 ) ;
	srgb.y = static_cast<unsigned char>( c.y*255 ) ;
	srgb.z = static_cast<unsigned char>( c.z*255 ) ;
	srgb.w = 255u ;
}

extern "C" __global__ void none( const float3* src, uchar4* dst, const int w, const int h ) {
	const int x = threadIdx.x+blockIdx.x*blockDim.x ;
	const int y = threadIdx.y+blockIdx.y*blockDim.y ;

	if ( x >= w )
		return ;
	if ( y >= h )
		return ;

	const int pix = x+w*y ;
	const float3 s = src[pix] ;
	const uchar4 d = make_uchar4(
		static_cast<unsigned char>( s.x*255 ),
		static_cast<unsigned char>( s.y*255 ),
		static_cast<unsigned char>( s.z*255 ), 255u ) ;
	dst[pix] = d ;
}

extern "C" __global__ void sRGB( const float3* src, uchar4* dst, const int w, const int h ) {
	const int x = threadIdx.x+blockIdx.x*blockDim.x ;
	const int y = threadIdx.y+blockIdx.y*blockDim.y ;

	if ( x >= w )
		return ;
	if ( y >= h )
		return ;

	const int pix = x+w*y ;
	sRGB( src[pix], dst[pix] ) ;
}

extern "C" __host__ void pp_none( const float3* src, uchar4* dst, const int w, const int h ) {
	const int blocks_x = ( w+32-1 )/32 ;
	const int blocks_y = ( h+32-1 )/32 ;

	none<<<dim3( blocks_x, blocks_y, 1 ), dim3( 32, 32, 1 )>>>( src, dst, w, h ) ;
}

extern "C" __host__ void pp_sRGB( const float3* src, uchar4* dst, const int w, const int h ) {
	const int blocks_x = ( w+32-1 )/32 ;
	const int blocks_y = ( h+32-1 )/32 ;

	sRGB<<<dim3( blocks_x, blocks_y, 1 ), dim3( 32, 32, 1 )>>>( src, dst, w, h ) ;
}
