// system includes
#include <iostream>
#include <vector>
#ifndef _MSC_VER
#include <getopt.h>
#else
#define strcasecmp _stricmp
#endif // _MSC_VER

// subsystem includes
// none

// local includes
// none

// file specific includes
#include "args.h"

#ifdef _MSC_VER
int optind = 0 ;
#endif // _MSC_VER

Args::Args( const int argc, char* const* argv ) noexcept( false ) {

#ifndef _MSC_VER

	int c, n = 0 ;

	const char*         s_opts   = "g:a:s:d:vtqhA:GSD:b" ;
	const struct option l_opts[] = {
		{ "geometry",          required_argument, 0, 'g' },
		{ "aspect-ratio",      required_argument, 0, 'a' },
		{ "samples-per-pixel", required_argument, 0, 's' },
		{ "trace-depth",       required_argument, 0, 'd' },
		{ "verbose",           no_argument,       &v_, 1 },
		{ "trace-sm",          no_argument,       &t_, 1 },
		{ "quiet",             no_argument,       &q_, 1 },
		{ "silent",            no_argument,       &q_, 1 },
		{ "help",              no_argument,       &h_, 1 },
		{ "usage",             no_argument,       &h_, 1 },
		{ "print-aov",         required_argument, 0, 'A' },
		{ "print-guides",      no_argument,       &G_, 1 },
		{ "print-statistics",  no_argument,       &S_, 1 },
		{ "apply-denoiser",    required_argument, 0, 'D' },
		{ "batch-mode",        required_argument, &b_, 1 },
		{ 0, 0, 0, 0 }
	} ;

	do {
		switch ( c = getopt_long( argc, argv, s_opts, l_opts, 0 ) ) {
			case 'g':
				{
					auto s = res_map.find( optarg ) ;
					if ( s == res_map.end() ) {
						sscanf( optarg, "%dx%d", &g_w_, &g_h_ ) ;
						if ( 1>g_w_ || 1>g_h_ ) {
							g_w_ = -1 ;
							g_h_ = -1 ;
						}
					} else { 
						g_w_ = abs( s->second.w ) ;
						g_h_ = abs( s->second.h ) ;
					}
				}
				break ;
			case 'a':
				float w, h ;
				sscanf( optarg, "%f:%f", &w, &h ) ;
				if ( g_w_>0 && w>0 && h>0 )
					g_h_ = static_cast<int>( static_cast<float>( g_w_ )*h/w+.5f ) ;
				break ;
			case 's':
				s_ = abs( atoi( optarg ) );
				break ;
			case 'd':
				d_ = abs( atoi( optarg ) ) ;
				break ;
			case 'v':
				v_ = 1 ;
				break ;
			case 't':
				t_ = 1 ;
				break ;
			case 'q':
				q_ = 1 ;
				break ;
			case 'h':
				h_  = 1 ;
				break ;
			case 'A':
				{
					std::vector<const char*> subopt{ optarg } ;
					for ( int i = 0 ; optarg[i] != '\0' ; i++ )
						if ( optarg[i] == ',' ) {
							optarg[i] = '\0' ; // terminate subopt
							subopt.push_back( &optarg[i+1] ) ;
						}
					for ( auto aov : subopt ) {
						auto s = aov_map.find( aov ) ;
						if ( s != aov_map.end() ) {
							if ( s->second == Aov::RPP ) A_rpp_ = Aov::RPP ;
						} else
							std::cerr << "rtwo: unknown argument for option A ignored -- " << aov << std::endl ;
					}
				}
				break ;
			case 'G':
				G_ = 1 ;
				break ;
			case 'S':
				S_ = 1 ;
				break ;
			case 'D':
				{
					auto s = dns_map.find( optarg ) ;
					if ( s != dns_map.end() )
						D_typ_ = s->second ;
					else
						std::cerr << "rtwo: unknown argument for option D ignored -- " << optarg << std::endl ;
				}
				break ;
			case 'b':
				b_ = 1 ;
				break ;
			case '?':
				throw std::invalid_argument( "try 'rtwo --help' for more information." ) ;
				break ;
			default: // -1
				break ;
		}
	} while ( c>-1 && MAXOPT>n++ ) ;

#else

	switch ( argc ) {
		case 1:
			break ;
		case 2:
			if ( ! strcasecmp( argv[1], "/h" ) || ! strcasecmp( argv[1], "/help" ) ) {
				h_ = 1 ;
				break ;
			}
			if ( strcasecmp( argv[1], "/b" ) || strcasecmp( argv[1], "/batch-mode" ) ) {
				b_ = 1 ;
				break ;
			}
		default:
			std::cerr << "rtwo: unknown options" << std::endl ;
			throw std::invalid_argument( "try 'rtwo /help' for more information." ) ;
	}

	optind = argc ;

#endif // _MSC_VER

}

int  Args::param_w( const int dEfault ) const { return 0>g_w_ ? dEfault : g_w_ ; }
int  Args::param_h( const int dEfault ) const { return 0>g_h_ ? dEfault : g_h_ ; }
int  Args::param_s( const int dEfault ) const { return 0>s_ ? dEfault : s_ ; }
int  Args::param_d( const int dEfault ) const { return 0>d_ ? dEfault : d_ ; }
Dns  Args::param_D( const Dns dEfault ) const { return D_typ_ == Dns::NONE ? dEfault : D_typ_ ; }

bool Args::flag_v()                     const { return v_>0 ; }
bool Args::flag_h()                     const { return h_>0 ; }
bool Args::flag_q()                     const { return q_>0 ; }
bool Args::flag_t()                     const { return t_>0 ; }
bool Args::flag_G()                     const { return G_>0 ; }
bool Args::flag_S()                     const { return S_>0 ; }
bool Args::flag_b()                     const { return b_>0 ; }

bool Args::flag_A( const Aov select )   const { return A_rpp_ == select ; }

void Args::usage() {
#ifndef _MSC_VER
	std::cerr << "Usage: rtwo [OPTION]... [FILE TRANSFORM]...\n\
  rtwo renders the final image from Pete Shirley's book Ray Tracing in\n\
  One Weekend using NVIDIA's OptiX Ray Tracing Engine and pipes the result\n\
  (PPM) to stdout for easy batch post-processing (e.g. ImageMagick).\n\
\n\
  If the host execs an X server (GLX enabled) as well, rtwo continuously\n\
  renders and displays results. A (rather) simple UI allows for basic\n\
  interactions.\n\
\n\
  Additional objects in FILE must conform to Wavefront OBJ and each have\n\
  a TRANSFOM. The format is sx:sy:sz:tx:ty:tz with floats for scaling\n\
  and translating in sx, sy, sz and tx, ty, tz respectively.\n\
\n\
Examples:\n\
  # render and convert image to PNG. Print statistical information on stderr.\n\
  rtwo -S | magick ppm:- rtwo.png\n\
\n\
  # output image and AOV (yields rtwo-0.png (image) and rtwo-1.png (RPP)).\n\
  rtwo --print-aov RPP | magick - rtwo.png\n\
\n\
  # improve AOV RPP contrast.\n\
  magick rtwo-1.png -auto-level -level 0%,25% rtwo-rpp.png\n\
\n\
  # apply denoiser and output guide layers (yields rtwo-0.png (normals),\n\
  # rtwo-1.png (albedos), and rtwo-2.png (image)).\n\
  rtwo --print-guides --apply-denoiser NAA | magick - rtwo.png\n\
\n\
Options:\n\
  -g, --geometry {<width>[x<height>]|RES}\n\
    Set image resolution to particular dimensions or common values predefined\n\
    by RES.\n\
\n\
    Prefedined resolutions for RES:\n\
      CGA    - 320x200      8:5\n\
      HVGA   - 480x320      3:2\n\
      VGA    - 640x480      4:3\n\
      WVGA   - 800x480      5:3\n\
      SVGA   - 800x600      4:3\n\
      XGA    - 1024x768     4:3\n\
      HD     - 1280x720    16:9 (default)\n\
      SXGA   - 1280x1024    5:4\n\
      UXGA   - 1600x1200    4:3\n\
      FullHD - 1920x1080   16:9\n\
      2K     - 2048x1080  ~17:9\n\
      QXGA   - 2048x1536    4:3\n\
      UWHD   - 2560x1080  ~21:9\n\
      WQHD   - 2560x1440   16:9\n\
      WQXGA  - 2560x1600    8:5\n\
      UWQHD  - 3440x1440  ~21:9\n\
      UHD-1  - 3840x2160   16:9\n\
      4K     - 4096x2160  ~17:9\n\
      5K-UW  - 5120x2160  ~21:9\n\
      5K     - 5120x2880   16:9\n\
      UHD-2  - 7680x4320   16:9\n\
\n\
    Default resolution is HD.\n\
\n\
  -s, --samples-per-pixel N\n\
    Use N samples per pixel (SPP). Default is 50.\n\
\n\
  -d, --trace-depth N\n\
    Trace N rays per sample (RPS). A minimum value of 1 means just trace\n\
    primary rays. Default is 16 or 50, depending on whether rtwo was\n\
    compiled for recursive or iterative ray tracing respectively.\n\
\n\
  -a, --aspect-ratio <width>:<height>\n\
    Set aspect ratio if not (implicitly) defined by -g, --geometry\n\
    options. Default is 16:9 (HD).\n\
\n\
  -v, --verbose\n\
    Print processing details on stderr. Has impacts on performance.\n\
\n\
  -t, --trace-sm\n\
    Print FSM events and state changes to stderr (debugging).\n\
    This option is not available in batch mode.\n\
\n\
  -q, --quiet, --silent\n\
    Omit result output on stdout. This option takes precedence over\n\
    -A, --print-aov and -G, --print-guides options.\n\
\n\
  -h, --help, --usage\n\
    Print this help. Takes precedence over any other options.\n\
\n\
  -A, --print-aov <AOV>[,...]\n\
    Print AOVs after image on stdout. Option not available in interactive\n\
    mode.\n\
\n\
    Available AOVs:\n\
      RPP (1) - Rays per pixel. Sort of `data' AOV (opposed to AOVs for\n\
                lighting or shading). Values add up total number of rays\n\
                traced for each pixel. Output format is PGM.\n\
\n\
  -G, --print-guides\n\
    Print denoiser guide layers before image on stdout. Available guide layers\n\
    depend on denoiser type: SMP/ none, NRM/ normals, ALB/ albedos, NAA/ both,\n\
    AOV/ both. Output format is PPM. Option not available in interactive mode.\n\
\n\
  -S, --print-statistics\n\
    Print statistical information on stderr.\n\
\n\
  -D, --apply-denoiser <TYP>\n\
    Enable denoiser in batch mode and apply type TYP after rendering with\n\
    1 SPP. To enable for interactive mode see UI functions section below.\n\
    Denoising in interactive mode applies to scene animation and while\n\
    changing camera position and direction, as well as zooming.\n\
    When finished there is a final still image rendered with SPP as given\n\
    by -s, --samples-per-pixels or default.\n\
\n\
    Available types for TYP:\n\
      SMP (1) - A simple type using OPTIX_DENOISER_MODEL_KIND_LDR. Feeds raw\n\
                RGB rendering output into denoiser and retrieves result.\n\
      NRM (2) - Simple type plus hit point normals.\n\
      ALB (3) - Simple type plus albedo values for hit points.\n\
      NAA (4) - Simple type plus normals and albedos.\n\
      AOV (5) - The NAA type using OPTIX_DENOISER_MODEL_KIND_AOV.\n\
                Might improve denoising result even if no AOVs provided.\n\
\n\
  -b, --batch-mode\n\
    Force batch mode even when running a GLX-enabled X server.\n\
\n\
" ;
#else
	std::cerr << "Usage: rtwo [/h|/help]\n\
  rtwo renders the final image from Pete Shirley's book Ray Tracing in\n\
  One Weekend using NVIDIA's OptiX Ray Tracing Engine and pipes the result\n\
  (PPM) to stdout for easy batch post-processing (e.g. ImageMagick).\n\
\n\
  If the host execs an X server (GLX enabled) as well, rtwo continuously\n\
  renders and displays results. A (rather) simple UI allows for basic\n\
  interactions.\n\
\n\
Example:\n\
  # render and convert image to PNG.\n\
  rtwo | magick ppm:- rtwo.png\n\
\n\
Options:\n\
\n\
  /h, /help\n\
    Print this help.\n\
\n\
  /b, /batch-mode\n\
    Force batch mode even when running a GLX-enabled X server.\n\
\n\
" ;
#endif // _MSC_VER
}

#ifdef MAIN

int main( int argc, char* argv[] ) {
	try {
		Args args( argc, argv ) ;

		if ( args.flag_h() ) {
			Args::usage() ;

			return 0 ;
		}

		std::cout << "geometry   : " << args.param_w( 4711 ) << "x" << args.param_h( 4711 ) << std::endl ;
		std::cout << "spp        : " << args.param_s( 4711 ) << std::endl ;
		std::cout << "depth      : " << args.param_d( 4711 ) << std::endl ;

		std::cout << "denoiser   : " << static_cast<int>( args.param_D( Dns::NONE ) ) << std::endl ;

		std::cout << "verbose    : " << ( args.flag_v() ? "set" : "not set" ) << std::endl ;
		std::cout << "quiet      : " << ( args.flag_q() ? "set" : "not set" ) << std::endl ;
		std::cout << "silent     : " << ( args.flag_q() ? "set" : "not set" ) << std::endl ;
		std::cout << "trace-sm   : " << ( args.flag_t() ? "set" : "not set" ) << std::endl ;
		std::cout << "guides     : " << ( args.flag_G() ? "set" : "not set" ) << std::endl ;
		std::cout << "statistics : " << ( args.flag_S() ? "set" : "not set" ) << std::endl ;
		std::cout << "batch-mode : " << ( args.flag_b() ? "set" : "not set" ) << std::endl ;

		std::cout << "aov RPP    : " << ( args.flag_A( Aov::RPP ) ? "set" : "not set" ) << std::endl ;
	} catch ( const std::invalid_argument& e ) {
		std::cerr << e.what() << std::endl ;

		return 1 ;
	}

	return 0 ;
}

#endif // MAIN
