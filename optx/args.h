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

class Args {
	public:
		Args( const int argc, char* const* argv ) noexcept( false ) ;

		int param_w( const int dEfault ) const ;
		int param_h( const int dEfault ) const ;
		int param_spp  ( const int dEfault ) const ;
		int param_depth( const int dEfault ) const ;

		bool flag_verbose() const ;
		bool flag_help() const ;
		bool flag_quiet() const ;
		bool flag_tracesm() const ;

		static void usage() ;

	private:
		int w_       = -1 ;
		int h_       = -1 ;
		int spp_     = -1 ;
		int depth_   = -1 ;
		int verbose_ =  0 ;
		int help_    =  0 ;
		int quiet_   =  0 ;
		int tracesm_ =  0 ;
} ;

#endif // ARGS_H
