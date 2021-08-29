#ifndef ARGS_H
#define ARGS_H

#include <map>

#define MAXOPT 32

typedef struct { int w; int h ; } res ;
static const std::map<std::string, res> res_map = {
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

enum class Aov { NONE, RPP } ;
static const std::string aov_name[] = { "none", "RPP" } ;
static const std::map<std::string, Aov> aov_map = {
	{ aov_name[static_cast<int>( Aov::RPP )], Aov::RPP }
} ;

enum class Dns { NONE, SMP, NRM, ALB, NAA, AOV } ;
static const std::string dns_name[] = { "none", "SMP", "NRM", "ALB", "NAA", "AOV" } ;
static const std::map<std::string, Dns> dns_map = {
	{ dns_name[static_cast<int>( Dns::SMP )], Dns::SMP },
	{ dns_name[static_cast<int>( Dns::NRM )], Dns::NRM },
	{ dns_name[static_cast<int>( Dns::ALB )], Dns::ALB },
	{ dns_name[static_cast<int>( Dns::NAA )], Dns::NAA },
	{ dns_name[static_cast<int>( Dns::AOV )], Dns::AOV }
} ;

class Args {
	public:
		Args( const int argc, char* const* argv ) noexcept( false ) ;

		int param_w( const int dEfault ) const ; // -g, --geometry <w>x<h>
		int param_h( const int dEfault ) const ;
		int param_s( const int dEfault ) const ; // -s, --samples-per-pixel
		int param_d( const int dEfault ) const ; // -t, --trace-depth

		Dns param_D( const Dns dEfault, const char** mnemonic = nullptr ) const ; // -D, --apply-denoiser

		bool flag_v() const ; // -v, --verbose
		bool flag_h() const ; // -h, --help
		bool flag_q() const ; // -q, --quiet
		bool flag_t() const ; // -t, --trace-sm
		bool flag_S() const ; // -S, --print-statistics

		bool flag_A( const Aov select ) const ; // -A, --print-aov

		static void usage() ;

	private:
		int g_w_   = -1 ;
		int g_h_   = -1 ;
		int s_     = -1 ;
		int d_     = -1 ;
		Dns D_typ_ = Dns::NONE ;

		int v_     =  0 ;
		int h_     =  0 ;
		int q_     =  0 ;
		int t_     =  0 ;
		int S_     =  0 ;
		Aov A_rpp_ = Aov::NONE ;
} ;

#endif // ARGS_H
