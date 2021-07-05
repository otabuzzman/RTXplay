#include <optix.h>
#include <optix_stubs.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h> // must follow glad.h

#include "simplesm.h"

#include "simpleui.h"

// finite state machine
static SimpleSM* simplesm ;
// event callback table
static void mousecliqCb( GLFWwindow* window, int key, int act, int mod )                       ;
static void mousemoveCb( GLFWwindow* window, double x, double y )                              ;
static void resizeCb   ( GLFWwindow* window, int w, int h )                                    ;
static void scrollCb   ( GLFWwindow* window, double x, double y )                              ;
static void keyCb      ( GLFWwindow* window, int key, int /*scancode*/, int act, int /*mod*/ ) ;

// GLSL sources of shaders
extern "C" const char vert_glsl[] ;
extern "C" const char frag_glsl[] ;

SimpleUI::SimpleUI( const std::string& name, LpGeneral& lp_general ) : lp_general_( lp_general ) {
	// initialize GLFW
	GLFW_CHECK( glfwInit() ) ;

	GLFW_CHECK( glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 ) ) ;
	GLFW_CHECK( glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 ) ) ;
	GLFW_CHECK( glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE ) ) ;

	const int w = lp_general_.image_w ;
	const int h = lp_general_.image_h ;
	GLFW_CHECK( window_ = glfwCreateWindow( w, h, name.data(), nullptr, nullptr ) ) ;
	GLFW_CHECK( glfwMakeContextCurrent( window_ ) ) ;

	// initialize GLAD
	if ( ! gladLoadGLLoader( (GLADloadproc) glfwGetProcAddress ) ) {
		std::ostringstream comment ;
		
		comment << "GLAD error: gladLoadGLLoader failed" << std::endl ;
		throw std::runtime_error( comment.str() ) ;
	}

	// initialize FSM
	simplesm = new SimpleSM( window_, &lp_general_ ) ;
	// initialize CBT
	GLFW_CHECK( glfwSetMouseButtonCallback( window_, mousecliqCb ) ) ;
	GLFW_CHECK( glfwSetCursorPosCallback  ( window_, mousemoveCb ) ) ;
	GLFW_CHECK( glfwSetWindowSizeCallback ( window_, resizeCb    ) ) ;
	GLFW_CHECK( glfwSetScrollCallback     ( window_, scrollCb    ) ) ;
	GLFW_CHECK( glfwSetKeyCallback        ( window_, keyCb       ) ) ;

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
			<< "link shader program_ failed: "
			<< log << std::endl ;
		throw std::runtime_error( comment.str() ) ;
	}
	GL_CHECK( uniform_ = glGetUniformLocation( program_, "t" ) ) ;

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
	GL_CHECK( glGenBuffers( 1, &pbo_ ) ) ;
	GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, pbo_ ) ) ;
	GL_CHECK( glBufferData(
		GL_ARRAY_BUFFER,
		sizeof( uchar4 )*w*h,
		nullptr,
		GL_STATIC_DRAW
		) ) ;

	// register pbo for CUDA
	CUDA_CHECK( cudaGraphicsGLRegisterBuffer( &glx_, pbo_, cudaGraphicsMapFlagsWriteDiscard ) ) ;
	GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) ) ;
}

SimpleUI::~SimpleUI() noexcept ( false ) {
	CUDA_CHECK( cudaGraphicsUnregisterResource( glx_ ) ) ;
	GL_CHECK( glDeleteTextures( 1, &tex_ ) ) ;
	GL_CHECK( glDeleteBuffers( 1, &pbo_ ) ) ;
	GL_CHECK( glDeleteVertexArrays( 1, &vao_ ) ) ;
	GL_CHECK( glDeleteBuffers( 1, &vbo_ ) ) ;
	GL_CHECK( glDeleteShader( v_shader_ ) ) ;
	GL_CHECK( glDeleteShader( f_shader_ ) ) ;
	GL_CHECK( glDeleteProgram( program_ ) ) ;
	GLFW_CHECK( glfwDestroyWindow( window_ ) ) ;
	GLFW_CHECK( glfwTerminate() ) ;
}

void SimpleUI::render( const OptixPipeline pipeline, const OptixShaderBindingTable& sbt ) {
	LpGeneral* lp_general = &lp_general_ ;
	size_t lp_general_size = sizeof( LpGeneral ), lp_general_image_size ;

	CUdeviceptr d_lp_general ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_lp_general ), sizeof( LpGeneral ) ) ) ;

	do {
		// launch pipeline
		CUstream cuda_stream ;
		CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;

		CUDA_CHECK( cudaGraphicsMapResources( 1, &glx_, cuda_stream ) ) ;
		CUDA_CHECK( cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void**>( &lp_general->image ), &lp_general_image_size, glx_ ) ) ;
		CUDA_CHECK( cudaMemcpy(
			reinterpret_cast<void*>( d_lp_general ),
			lp_general,
			lp_general_size,
			cudaMemcpyHostToDevice
			) ) ;
		const int w = lp_general->image_w ;
		const int h = lp_general->image_h ;
		OPTX_CHECK( optixLaunch(
			pipeline,
			cuda_stream,
			d_lp_general,
			lp_general_size,
			&sbt,
			w/*x*/, h/*y*/, 1/*z*/ ) ) ;
		CUDA_CHECK( cudaDeviceSynchronize() ) ;

		CUDA_CHECK( cudaGraphicsUnmapResources( 1, &glx_, cuda_stream ) ) ;

		CUDA_CHECK( cudaStreamDestroy( cuda_stream ) ) ;

		// display result
		GL_CHECK( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) ) ;

		GL_CHECK( glClearColor( .333f, .525f, .643f, 1.f ) ) ;
		GL_CHECK( glClear( GL_COLOR_BUFFER_BIT ) ) ;

		GL_CHECK( glUseProgram( program_ ) ) ;
		GL_CHECK( glUniform1i( uniform_ , 0 ) ) ;

		GL_CHECK( glActiveTexture( GL_TEXTURE0 ) ) ;
		GL_CHECK( glBindTexture( GL_TEXTURE_2D, tex_ ) ) ;
		GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo_ ) ) ;
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
		GLFW_CHECK( glfwWaitEvents() ) ;
	} while ( ! glfwWindowShouldClose( window_ ) ) ;
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

static void scrollCb( GLFWwindow* /*window*/, double /*x*/, double /*y*/ ) {
	simplesm->transition( Event::SCR ) ;
}

static void keyCb( GLFWwindow* /*window*/, int key, int /*scancode*/, int act, int /*mod*/ ) {
	if ( act != GLFW_PRESS )
		return ;
	switch ( key ) {
		case GLFW_KEY_Z:
			simplesm->transition( Event::ZOM ) ; break ;
		case GLFW_KEY_B:
			simplesm->transition( Event::BLR ) ; break ;
		case GLFW_KEY_F:
			simplesm->transition( Event::FOC ) ; break ;
		case GLFW_KEY_Q:
		case GLFW_KEY_ESCAPE:
			simplesm->transition( Event::RET ) ; break ;
	}
}
