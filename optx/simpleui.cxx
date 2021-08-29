#include <chrono>

#include <optix.h>
#include <optix_stubs.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h> // must follow glad.h

#include "simpleui.h"

// common globals
extern Args*     args ;
extern LpGeneral lp_general ;

// missing in GLFW
void glfwSetScroll( GLFWwindow* window, const double xscroll, const double yscroll ) ;
void glfwGetScroll( GLFWwindow* window, double* xscroll, double* yscroll ) ;
static double xscroll_ = 0. ;
static double yscroll_ = 0. ;

// finite state machine
static SimpleSM* simplesm ;
static SmParam   smparam  ;
// event callback table
static void mousecliqCb( GLFWwindow* window, int key, int act, int mod )                       ;
static void mousemoveCb( GLFWwindow* window, double x, double y )                              ;
static void resizeCb   ( GLFWwindow* window, int w, int h )                                    ;
static void scrollCb   ( GLFWwindow* window, double x, double y )                              ;
static void keyCb      ( GLFWwindow* window, int key, int /*scancode*/, int act, int /*mod*/ ) ;

// GLSL sources of shaders
extern "C" const char vert_glsl[] ;
extern "C" const char frag_glsl[] ;

// post processing
extern "C" void pp_none( const float3* src, uchar4* dst, const int w, const int h ) ;
extern "C" void pp_sRGB( const float3* src, uchar4* dst, const int w, const int h ) ;

SimpleUI::SimpleUI( const std::string& name ) {
	// initialize GLFW
	GLFW_CHECK( glfwInit() ) ;

	GLFW_CHECK( glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 ) ) ;
	GLFW_CHECK( glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 ) ) ;
	GLFW_CHECK( glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE ) ) ;

	const int w = lp_general.image_w ;
	const int h = lp_general.image_h ;
	GLFW_CHECK( window_ = glfwCreateWindow( w, h, name.data(), nullptr, nullptr ) ) ;
	GLFW_CHECK( glfwMakeContextCurrent( window_ ) ) ;

	// initialize GLAD
	if ( ! gladLoadGLLoader( (GLADloadproc) glfwGetProcAddress ) ) {
		std::ostringstream comment ;
		
		comment << "GLAD error: gladLoadGLLoader failed" << std::endl ;
		throw std::runtime_error( comment.str() ) ;
	}

	// compile vertex shader
	GL_CHECK( v_shader_ = glCreateShader( GL_VERTEX_SHADER ) ) ;
	const GLchar* src = reinterpret_cast<const GLchar*>( vert_glsl ) ;
	GL_CHECK( glShaderSource( v_shader_, 1, &src, nullptr ) ) ;
	GL_CHECK( glCompileShader( v_shader_ ) ) ;
	GLint success = 0 ;
	GL_CHECK( glGetShaderiv( v_shader_, GL_COMPILE_STATUS, &success ) ) ;
	if ( success == GL_FALSE ) {
		char log[512] ;
		glGetShaderInfoLog( v_shader_, 512, nullptr, log ) ;
		glDeleteShader( v_shader_ ) ;
		std::ostringstream comment ;
		comment
			<< "vertex shader compilation failed: "
			<< log << std::endl ;
		throw std::runtime_error( comment.str() ) ;
	}

	// compile fragment shader
	GL_CHECK( f_shader_ = glCreateShader( GL_FRAGMENT_SHADER ) ) ;
	src = reinterpret_cast<const GLchar*>( frag_glsl ) ;
	GL_CHECK( glShaderSource( f_shader_, 1, &src, nullptr ) ) ;
	GL_CHECK( glCompileShader( f_shader_ ) ) ;
	GL_CHECK( glGetShaderiv( f_shader_, GL_COMPILE_STATUS, &success ) ) ;
	if ( success == GL_FALSE ) {
		char log[512] ;
		glGetShaderInfoLog( f_shader_, 512, nullptr, log ) ;
		glDeleteShader( f_shader_ ) ;
		std::ostringstream comment ;
		comment
			<< "fragment shader compilation failed: "
			<< log << std::endl ;
		throw std::runtime_error( comment.str() ) ;
	}

	// create program
	GL_CHECK( program_ = glCreateProgram() ) ;
	GL_CHECK( glAttachShader( program_, v_shader_ ) ) ;
	GL_CHECK( glAttachShader( program_, f_shader_ ) ) ;
	GL_CHECK( glLinkProgram( program_ ) ) ;
	GL_CHECK( glDetachShader( program_, v_shader_ ) ) ;
	GL_CHECK( glDetachShader( program_, f_shader_ ) ) ;
	GL_CHECK( glGetProgramiv( program_, GL_LINK_STATUS, &success ) ) ;
	if ( success == GL_FALSE ) {
		char log[512] ;
		glGetProgramInfoLog( v_shader_, 512, nullptr, log ) ;
		glDeleteProgram( program_ ) ;
		glDeleteShader( f_shader_ ) ;
		glDeleteShader( v_shader_ ) ;
		std::ostringstream comment ;
		comment
			<< "link shader program failed: "
			<< log << std::endl ;
		throw std::runtime_error( comment.str() ) ;
	}
	GL_CHECK( uniform_ = glGetUniformLocation( program_, "t" ) ) ;

	// allocate render buffer
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.rawRGB ), sizeof( float3 )*w*h ) ) ;
	// allocate rays per pixel (rpp) buffer
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.rpp ), sizeof( unsigned int )*w*h ) ) ;

	// setup vertex buffer object
	GL_CHECK( glGenVertexArrays( 1, &vao_ ) ) ;
	GL_CHECK( glBindVertexArray( vao_ ) ) ;
	// vertices of two triangles forming a rectangle and spanning the viewport (NDC)
	static const GLfloat vces[] = {
		-1.f, -1.f, .0f,
		1.f, -1.f, .0f,
		-1.f,  1.f, .0f,

		-1.f,  1.f, .0f,
		1.f, -1.f, .0f,
		1.f,  1.f, .0f,
	} ;
	GL_CHECK( glGenBuffers( 1, &vbo_ ) ) ;
	GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, vbo_ ) ) ;
	GL_CHECK( glBufferData(
		GL_ARRAY_BUFFER,
		sizeof( vces ),
		vces,
		GL_STATIC_DRAW
		) ) ;
	GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) ) ;

	// setup texture (image)
	GL_CHECK( glGenTextures( 1, &tex_ ) ) ;
	GL_CHECK( glBindTexture( GL_TEXTURE_2D, tex_ ) ) ;
	GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) ) ;
	GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) ) ;
	GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) ) ;
	GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE ) ) ;

	// create pixel (image) buffer object
	GL_CHECK( glGenBuffers( 1, &smparam.pbo ) ) ;
	GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, smparam.pbo ) ) ;
	GL_CHECK( glBufferData(
		GL_ARRAY_BUFFER,
		sizeof( uchar4 )*w*h,
		nullptr,
		GL_STATIC_DRAW
		) ) ;

	// register pbo for CUDA OpenGL interop
	CUDA_CHECK( cudaGraphicsGLRegisterBuffer( &smparam.glx, smparam.pbo, cudaGraphicsMapFlagsWriteDiscard ) ) ;
	GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) ) ;

	// initialize FSM
	glfwSetWindowUserPointer( window_, &smparam ) ;
	simplesm = new SimpleSM( window_ ) ;
	// initialize CBT
	GLFW_CHECK( glfwSetMouseButtonCallback( window_, mousecliqCb ) ) ;
	GLFW_CHECK( glfwSetCursorPosCallback  ( window_, mousemoveCb ) ) ;
	GLFW_CHECK( glfwSetWindowSizeCallback ( window_, resizeCb    ) ) ;
	GLFW_CHECK( glfwSetScrollCallback     ( window_, scrollCb    ) ) ;
	GLFW_CHECK( glfwSetKeyCallback        ( window_, keyCb       ) ) ;
}

SimpleUI::~SimpleUI() noexcept ( false ) {
	delete simplesm ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.rawRGB ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.rpp ) ) ) ;
	CUDA_CHECK( cudaGraphicsUnregisterResource( smparam.glx ) ) ;
	GL_CHECK( glDeleteTextures( 1, &tex_ ) ) ;
	GL_CHECK( glDeleteBuffers( 1, &smparam.pbo ) ) ;
	GL_CHECK( glDeleteVertexArrays( 1, &vao_ ) ) ;
	GL_CHECK( glDeleteBuffers( 1, &vbo_ ) ) ;
	GL_CHECK( glDeleteShader( v_shader_ ) ) ;
	GL_CHECK( glDeleteShader( f_shader_ ) ) ;
	GL_CHECK( glDeleteProgram( program_ ) ) ;
	GLFW_CHECK( glfwDestroyWindow( window_ ) ) ;
	GLFW_CHECK( glfwTerminate() ) ;
}

void SimpleUI::render( const OptixPipeline pipeline, const OptixShaderBindingTable& sbt ) {
	size_t lp_general_size = sizeof( LpGeneral ), lp_general_image_size ;

	CUdeviceptr d_lp_general ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_lp_general ), sizeof( LpGeneral ) ) ) ;

	do {
		auto t0 = std::chrono::high_resolution_clock::now() ;
		// launch pipeline
		CUstream cuda_stream ;
		CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;

		CUDA_CHECK( cudaGraphicsMapResources( 1, &smparam.glx, cuda_stream ) ) ;
		CUDA_CHECK( cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void**>( &lp_general.image ), &lp_general_image_size, smparam.glx ) ) ;
		CUDA_CHECK( cudaMemcpy(
			reinterpret_cast<void*>( d_lp_general ),
			&lp_general,
			lp_general_size,
			cudaMemcpyHostToDevice
			) ) ;

		const int w = lp_general.image_w ;
		const int h = lp_general.image_h ;

		auto t1 = std::chrono::high_resolution_clock::now() ;
		OPTX_CHECK( optixLaunch(
			pipeline,
			cuda_stream,
			d_lp_general,
			lp_general_size,
			&sbt,
			w/*x*/, h/*y*/, 1/*z*/ ) ) ;
		CUDA_CHECK( cudaDeviceSynchronize() ) ;
		auto t2 = std::chrono::high_resolution_clock::now() ;

		if ( args->flag_S() ) { // output statistics
			long long dt = std::chrono::duration_cast<std::chrono::milliseconds>( t2-t1 ).count() ;
			std::vector<unsigned int> rpp ;
			rpp.resize( w*h ) ;
			CUDA_CHECK( cudaMemcpy(
				reinterpret_cast<void*>( rpp.data() ),
				lp_general.rpp,
				w*h*sizeof( unsigned int ),
				cudaMemcpyDeviceToHost
				) ) ;
			long long sr = 0 ; for ( auto const& c : rpp ) sr = sr+c ; // accumulate rays per pixel
			fprintf( stderr, "%9u %12llu %4llu (pixels, rays, milliseconds)", w*h, sr, dt ) ;
		}

		// apply denoiser
		if ( smparam.dns_exec && smparam.dns_type != Dns::NONE )
			smparam.denoiser->beauty( lp_general.rawRGB ) ;

		// post processing
		pp_sRGB( lp_general.rawRGB, lp_general.image, w, h ) ;

		CUDA_CHECK( cudaGraphicsUnmapResources( 1, &smparam.glx, cuda_stream ) ) ;

		CUDA_CHECK( cudaStreamDestroy( cuda_stream ) ) ;

		// display result
		GL_CHECK( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) ) ;
		glViewport( 0, 0, w, h ) ;

		GL_CHECK( glClearColor( .333f, .525f, .643f, 1.f ) ) ;
		GL_CHECK( glClear( GL_COLOR_BUFFER_BIT ) ) ;

		GL_CHECK( glUseProgram( program_ ) ) ;
		GL_CHECK( glUniform1i( uniform_ , 0 ) ) ;

		GL_CHECK( glActiveTexture( GL_TEXTURE0 ) ) ;
		GL_CHECK( glBindTexture( GL_TEXTURE_2D, tex_ ) ) ;
		GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, smparam.pbo ) ) ;
		GL_CHECK( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) ) ;
		GL_CHECK( glTexImage2D(
			GL_TEXTURE_2D,
			0,                // for mipmap level
			GL_RGBA8,         // texture color components
			w,
			h,
			0,
			GL_RGBA,          // pixel data format
			GL_UNSIGNED_BYTE, // pixel data type
			nullptr           // data in GL_PIXEL_UNPACK_BUFFER (pbo)
			) ) ;
		GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 ) ) ;

		GL_CHECK( glEnableVertexAttribArray( 0 ) ) ;
		GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, vbo_ ) ) ;
		GL_CHECK( glVertexAttribPointer(
			0,        // attribute index
			3,        // attribute components
			GL_FLOAT, // attribute components type
			GL_FALSE, // normalize fix-point
			0,        // stride between consecutive attributes
			nullptr   // data in GL_ARRAY_BUFFER (vbo)
			) ) ;

		/*** apply gamma correction if necessary
		GL_CHECK( glEnable( GL_FRAMEBUFFER_SRGB ) ) ;
		***/
		GL_CHECK( glDrawArrays( GL_TRIANGLES, 0, 6 ) ) ;
		GL_CHECK( glDisableVertexAttribArray( 0 ) ) ;
		/***
		GL_CHECK( glDisable( GL_FRAMEBUFFER_SRGB ) ) ;
		***/

		GLFW_CHECK( glfwSwapBuffers( window_ ) ) ;
		if ( smparam.anm_exec ) {
			// rotate eye around y (WCS)
			Camera* camera = &lp_general.camera ;
			const float3 eye = camera->eye() ;
			const float3 pat = camera->pat() ;
			const float  len = V::len( eye-pat ) ;
			Paddle paddle( eye, pat, camera->vup() ) ;

			paddle.start( 0, 0 ) ;
			camera->eye( pat+len*paddle.move( -1, 0 ) ) ;

			GLFW_CHECK( glfwPollEvents() ) ;
		} else
			GLFW_CHECK( glfwWaitEvents() ) ;

		if ( args->flag_S() ) { // output statistics
			auto t3 = std::chrono::high_resolution_clock::now() ;
			long long dt = std::chrono::duration_cast<std::chrono::milliseconds>( t3-t0 ).count() ;
			fprintf( stderr, " %6.2f fps\n", 1000.f/dt ) ;
		}
	} while ( ! glfwWindowShouldClose( window_ ) ) ;

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_lp_general ) ) ) ;
}

// event callback table
static void mousecliqCb( GLFWwindow* /*window*/, int key, int act, int /*mod*/ ) {
	if ( act == GLFW_PRESS   && key == GLFW_MOUSE_BUTTON_LEFT )  { simplesm->transition( Event::POS ) ; return ; }
	if ( act == GLFW_PRESS   && key == GLFW_MOUSE_BUTTON_RIGHT ) { simplesm->transition( Event::DIR ) ; return ; }
	if ( act == GLFW_RELEASE && key == GLFW_MOUSE_BUTTON_LEFT )  { simplesm->transition( Event::RET ) ; return ; }
	if ( act == GLFW_RELEASE && key == GLFW_MOUSE_BUTTON_RIGHT ) { simplesm->transition( Event::RET ) ; return ; }
}

static void mousemoveCb( GLFWwindow* /*window*/, double /*x*/, double /*y*/ ) {
	simplesm->transition( Event::MOV ) ;
}

static void resizeCb( GLFWwindow* /*window*/, int /*w*/, int /*h*/ ) {
	simplesm->transition( Event::RSZ ) ;
}

static void scrollCb( GLFWwindow* window, double x, double y ) {
	glfwSetScroll( window, x, y ) ;
	simplesm->transition( Event::SCR ) ;
}

static void keyCb( GLFWwindow* /*window*/, int key, int /*scancode*/, int act, int /*mod*/ ) {
	if ( act != GLFW_PRESS )
		return ;
	switch ( key ) {
		case GLFW_KEY_A:
			simplesm->transition( Event::ANM ) ; break ;
		case GLFW_KEY_B:
			simplesm->transition( Event::BLR ) ; break ;
		case GLFW_KEY_D:
			simplesm->transition( Event::DNS ) ; break ;
		case GLFW_KEY_F:
			simplesm->transition( Event::FOC ) ; break ;
		case GLFW_KEY_Z:
			simplesm->transition( Event::ZOM ) ; break ;
		case GLFW_KEY_ESCAPE:
			simplesm->transition( Event::RET ) ; break ;
	}
}

void glfwSetScroll( GLFWwindow* /*window*/, const double x, const double y ) {
	xscroll_ = x ;
	yscroll_ = y ;
}

void glfwGetScroll( GLFWwindow* /*window*/, double* x, double* y ) {
	*x = xscroll_ ;
	*y = yscroll_ ;
}

void SimpleUI::usage() {
	std::cerr << "UI functions:\n\
  window resize        - change viewport dimensions\n\
  left button + move   - change camera position\n\
  right button + move  - change camera direction\n\
  right button + scoll - roll camera\n\
  'a' key              - toggle scene animation\n\
  'b' key              - enter blur mode (<ESC> to leave)\n\
      scroll (wheel)   - change defocus blur\n\
  'd' key              - enable/ disable denoising and select\n\
                         type (loop)\n\
  'f' key              - enter focus mode (<ESC> to leave)\n\
      scroll (wheel)   - change aperture\n\
  'z' key              - enter zoom mode (<ESC> to leave)\n\
      scroll (wheel)   - zoom in and out\n\
  <ESC> key            - leave RTWO\n\
\n\
" ;
}
