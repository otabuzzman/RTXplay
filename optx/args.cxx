#include <iostream>
#include <vector>
#include <getopt.h>

#include "args.h"

Args::Args( const int argc, char* const* argv ) noexcept( false ) {
	int c, n = 0 ;

	const char*         s_opts   = "g:a:s:d:vtqhA:SD:" ;
	const struct option l_opts[] = {
		{ "geometry",          required_argument, 0, 'g' },
		{ "aspect-ratio",      required_argument, 0, 'a' },
		{ "samples-per-pixel", required_argument, 0, 's' },
		{ "trace-depth",       required_argument, 0, 'd' },
		{ "verbose",           no_argument, &verbose_, 1 },
		{ "trace-sm",          no_argument, &tracesm_, 1 },
		{ "quiet",             no_argument, &quiet_,   1 },
		{ "silent",            no_argument, &quiet_,   1 },
		{ "help",              no_argument, &help_,    1 },
		{ "usage",             no_argument, &help_,    1 },
		{ "print-aov",         required_argument, 0, 'A' },
		{ "print-statistics",  no_argument, &statinf_, 1 },
		{ "apply-denoiser",    required_argument, 0, 'D' },
		{ 0, 0, 0, 0 }
	} ;

	do {
		switch ( c = getopt_long( argc, argv, s_opts, l_opts, 0 ) ) {
			case 'g':
				{
					auto s = res_.find( optarg ) ;
					if ( s == res_.end() ) {
						sscanf( optarg, "%dx%d", &w_, &h_ ) ;
						if ( 1>w_ || 1>h_ ) {
							w_ = -1 ;
							h_ = -1 ;
						}
					} else { 
						w_ = s->second.w ;
						h_ = s->second.h ;
					}
				}
				break ;
			case 'a':
				float w, h ;
				sscanf( optarg, "%f:%f", &w, &h ) ;
				if ( w_>0 && w>0 && h>0 )
					h_ = static_cast<int>( static_cast<float>( w_ )*h/w+.5f ) ;
				break ;
			case 's':
				spp_ = atoi( optarg ) ;
				break ;
			case 'd':
				depth_ = atoi( optarg ) ;
				break ;
			case 'v':
				verbose_ = 1 ;
				break ;
			case 't':
				tracesm_ = 1 ;
				break ;
			case 'q':
				quiet_ = 1 ;
				break ;
			case 'h':
				help_  = 1 ;
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
						auto s = aov_.find( aov ) ;
						if ( s != aov_.end() ) {
							switch ( s->second ) {
								case AOV_RPP:
									aov_rpp_ = 1 ;
									break ;
							}
						} else
							std::cerr << argv[0] << ": unknown argument for option A ignored -- " << aov << std::endl ;
					}
				}
				break ;
			case 'S':
				statinf_ = 1 ;
				break ;
			case 'D':
				{
					auto s = dns_.find( optarg ) ;
					if ( s != dns_.end() ) {
						switch ( s->second ) {
							case DNS_SMP:
							case DNS_NRM:
							case DNS_ALB:
							case DNS_NAA:
							case DNS_AOV:
								denoiser_ = s->second ;
								break ;
						}
					} else
						std::cerr << argv[0] << ": unknown argument for option D ignored -- " << optarg << std::endl ;
				}
				break ;
			case '?':
				throw std::invalid_argument( "try 'rtwo --help' for more information." ) ;
				break ;
			default: // -1
				break ;
		}
	} while ( c>-1 && MAXOPT>n++ ) ;
}

int Args::param_w( const int dEfault )        const { return 0>w_ ? dEfault : w_ ; }
int Args::param_h( const int dEfault )        const { return 0>h_ ? dEfault : h_ ; }
int Args::param_spp  ( const int dEfault )    const { return 0>spp_   ? dEfault : spp_   ; }
int Args::param_depth( const int dEfault )    const { return 0>depth_ ? dEfault : depth_ ; }
int Args::param_denoiser( const int dEfault ) const { return 0>denoiser_ ? dEfault : denoiser_ ; }

bool Args::flag_verbose()  const { return verbose_>0 ; }
bool Args::flag_help()     const { return help_>0    ; }
bool Args::flag_quiet()    const { return quiet_>0   ; }
bool Args::flag_tracesm()  const { return tracesm_>0 ; }
bool Args::flag_statinf()  const { return statinf_>0 ; }

bool Args::flag_aov_rpp()  const { return aov_rpp_ != AOV_NONE ; }

void Args::usage() {
	std::cerr << "Usage: rtwo [OPTION...]\n\
  `rtwo´ renders the image from Pete Shirley's Ray Tracing in One Weekend\n\
  tutorial using NVIDIA's OptiX Ray Tracing Engine and pipes the result\n\
  (PPM) to stdout for easy batch post-processing (e.g. ImageMagick).\n\
\n\
  If the host execs an X server (GLX enabled) as well, `rtwo´ continuously\n\
  renders and displays results. A (rather) simple UI allows for basic\n\
  interactions.\n\
\n\
Examples:\n\
  # render and convert image to PNG. Print statistical information.\n\
  rtwo -S | magick ppm:- rtwo.png\n\
\n\
  # convert image and AOV (yields rtwo-0.png (image), rtwo-1.png (RPP)).\n\
  rtwo --print-aov RPP | magick - rtwo.png\n\
\n\
  # improve AOV RPP contrast.\n\
  magick rtwo-1.png -auto-level -level 0%,25% rtwo-rpp.png\n\
\n\
Options:\n\
  -g, --geometry {<width>[x<height>]|RES}\n\
    Set image resolution to particular dimensions or substitute RES with\n\
    a predefined value. In case of X server exec'ing, this option defines\n\
    initial values that will change on resizing the window.\n\
\n\
    Prefedined values for RES:\n\
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
  -s, --samples-per-pixel n\n\
    Use n samples per pixel (SPP). Default is 50.\n\
\n\
  -d, --trace-depth n\n\
    Trace n rays per sample (RPS). A minimum value of 1 means just trace\n\
    primary rays. Default is 16 or 50, depending on whether `rtwo´ was\n\
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
    -A, --print-aov options.\n\
\n\
  -h, --help, --usage\n\
    Print this help. Takes precedence over any other options\n\
\n\
  -A, --print-aov <AOV>[,...]\n\
    Print AOVs after image on stdout. Print all if no AOVs given (order as\n\
    in list below) or pick particular AOVs (order as given).\n\
\n\
    Available AOVs:\n\
      RPP (Rays per pixel) - Sort of AOV. Pixel values sum up total number of\n\
                             rays that have been traced by each sample (PGM).\n\
\n\
    This option is not available in interactive mode.\n\
\n\
  -S, --print-statistics\n\
    Print statistical information on stderr.\n\
\n\
  -D, --apply-denoiser <KND>\n\
    Apply denoiser kind KND after rendering using 1 SPP. In interactive mode\n\
    denoising is applied in scene animation and while moving camera position\n\
    and direction, zooming, etc. When finished there is a final still\n\
    rendering with SPP as given by --samples-per-pixels or default.\n\
    Output in batch mode is a denoised image rendered with 1 SPP.\n\
\n\
    Available denoiser for KND:\n\
      SMP - A simple type using OPTIX_DENOISER_MODEL_KIND_LDR. Feeds raw RGB\n\
            rendering output into denoiser and retrieves result.\n\
      NRM - Simple type plus hit point normals.\n\
      ALB - Simple type plus albedo values for hit points.\n\
      NAA - Simple type plus normals and albedos.\n\
      AOV - The NAA type using OPTIX_DENOISER_MODEL_KIND_AOV. Might improve\n\
            denoising result even if no AOVs provided\n\
\n\
" ;
}

#ifdef MAIN

int main( int argc, char* argv[] ) {
	try {
		Args args( argc, argv ) ;

		if ( args.flag_help() ) {
			Args::usage() ;

			return 0 ;
		}

		std::cout << "geometry   : " << args.param_w       ( 4711 )     << "x" << args.param_h( 4711 ) << std::endl ;
		std::cout << "spp        : " << args.param_spp     ( 4711 )     << std::endl ;
		std::cout << "depth      : " << args.param_depth   ( 4711 )     << std::endl ;
		std::cout << "denoiser   : " << args.param_denoiser( DNS_NONE ) << std::endl ;

		std::cout << "verbose    : " << ( args.flag_verbose()  ? "set" : "not set" ) << std::endl ;
		std::cout << "help       : " << ( args.flag_help()     ? "set" : "not set" ) << std::endl ;
		std::cout << "quiet      : " << ( args.flag_quiet()    ? "set" : "not set" ) << std::endl ;
		std::cout << "silent     : " << ( args.flag_quiet()    ? "set" : "not set" ) << std::endl ;
		std::cout << "trace-sm   : " << ( args.flag_tracesm()  ? "set" : "not set" ) << std::endl ;
		std::cout << "statistics : " << ( args.flag_statinf()  ? "set" : "not set" ) << std::endl ;

		std::cout << "aov RPP    : " << ( args.flag_aov_rpp()  ? "set" : "not set" ) << std::endl ;
	} catch ( const std::invalid_argument& e ) {
		std::cerr << e.what() << std::endl ;

		return 1 ;
	}

	return 0 ;
}

#endif // MAIN
