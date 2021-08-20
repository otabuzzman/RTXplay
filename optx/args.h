#ifndef ARGS_H
#define ARGS_H

#include <map>

#define MAXOPT 32

typedef struct { int w; int h ; } res ;
static const std::map<std::string, res> res_ = {
	{ "CGA",    {  320,  200 } },
	{ "HVGA",   {  480,  320 } },
	{ "VGA",    {  640,  480 } },
	{ "WVGA",   {  800,  480 } },
	{ "SVGA",   {  800,  600 } },
	{ "XGA",    { 1024,  768 } },
	{ "HD",     { 1280,  720 } },
	{ "SXGA",   { 1280, 1024 } },
	{ "UXGA",   { 1600, 1200 } },
	{ "FULLHD", { 1920, 1080 } },
	{ "2K",     { 2048, 1080 } },
	{ "QXGA",   { 2048, 1536 } },
	{ "UWHD",   { 2560, 1080 } },
	{ "WQHD",   { 2560, 1440 } },
	{ "WQXGA",  { 2560, 1600 } },
	{ "UWQHD",  { 3440, 1440 } },
	{ "UHD-1",  { 3840, 2160 } },
	{ "4K",     { 4096, 2160 } },
	{ "5K-UW",  { 5120, 2160 } },
	{ "5K",     { 5120, 2880 } },
	{ "UHD-2",  { 7680, 4320 } }
} ;

#define AOV_NONE 0
#define AOV_RPP  1
static const std::map<std::string, int> aov_ = {
	{ "NON", AOV_NONE },
	{ "RPP", AOV_RPP }
} ;

#define DNS_NONE 0
#define DNS_SMP  1
#define DNS_NRM  2
#define DNS_ALB  3
#define DNS_NAA  4
static const std::map<std::string, int> dns_ = {
	{ "NON", DNS_NONE },
	{ "SMP", DNS_SMP },
	{ "NRM", DNS_NRM },
	{ "ALB", DNS_ALB },
	{ "NAA", DNS_NAA }
} ;

class Args {
	public:
		Args( const int argc, char* const* argv ) noexcept( false ) ;

		int param_w( const int dEfault )        const ;
		int param_h( const int dEfault )        const ;
		int param_spp  ( const int dEfault )    const ;
		int param_depth( const int dEfault )    const ;
		int param_denoiser( const int dEfault ) const ;

		bool flag_verbose() const ;
		bool flag_help()    const ;
		bool flag_quiet()   const ;
		bool flag_tracesm() const ;
		bool flag_statinf() const ;

		bool flag_aov_rpp() const ;

		bool flag_denoiser() const ; // pseudo flag

		static void usage() ;

	private:
		int w_       = -1 ;
		int h_       = -1 ;
		int spp_     = -1 ;
		int depth_   = -1 ;
		int denoiser_  = DNS_NONE ;

		int verbose_ =  0 ;
		int help_    =  0 ;
		int quiet_   =  0 ;
		int tracesm_ =  0 ;
		int statinf_ =  0 ;

		int aov_rpp_ =  AOV_NONE ;
} ;

#endif // ARGS_H
