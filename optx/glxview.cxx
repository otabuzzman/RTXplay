// system includes
#include <iostream>
#include <sstream>

// subsystem includes
// GLAD
#include <glad/glad.h>
// GLFW
#include <GLFW/glfw3.h>
// CUDA
#ifdef __NVCC__
#include <cuda.h>
#include <cuda_gl_interop.h> // must follow glad.h
#endif // __NVCC__

// local includes
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "util.h"

// file specific includes
// none

// GLSL sources of shaders
extern "C" const char vert_glsl[] ;
extern "C" const char frag_glsl[] ;

static double aspratio = 0. ;

static void onkey( GLFWwindow* window, const int keyname, const int keycode, const int keyact, const int keymod ) {
	GLFW_CHECK( glfwSetWindowShouldClose( window, GLFW_TRUE ) ) ;
}
static void onresize( GLFWwindow* window, const int w, const int h ) {
	const double r = static_cast<double>( w )/h ;

	// new aspect ratio more landscape than image thus w follows h
	if ( r>aspratio )
		GL_CHECK( glViewport( 0, 0, h*aspratio, h ) ) ;
	// new aspect ratio less landscape than image thus h follows w
	else
		GL_CHECK( glViewport( 0, 0, w, w/aspratio ) ) ;
}

int main( const int argc, const char** argv ) {
	unsigned char* image ;
	int w, h, c ;

	// load image from file
	if ( 2>argc )
		return( 58 ) ;
	if ( ( image = stbi_load( argv[1], &w, &h, &c, 0 ) ) == nullptr ) {
		std::cerr << "STBI error: stbi_load failed" << std::endl ;

		return( 1 ) ;
	}
	aspratio = static_cast<double>( w )/h ;

	try {
		// initialize
		GLFWwindow* window ;
		{
			// GLFW
			GLFW_CHECK( glfwInit() ) ;

			GLFW_CHECK( glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 ) ) ;
			GLFW_CHECK( glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 ) ) ;
			GLFW_CHECK( glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE ) ) ;

			GLFW_CHECK( window = glfwCreateWindow( w, h, "RTWO", nullptr, nullptr ) ) ;
			GLFW_CHECK( glfwMakeContextCurrent( window ) ) ;

			GLFW_CHECK( glfwSetKeyCallback            ( window, onkey ) ) ;
			GLFW_CHECK( glfwSetFramebufferSizeCallback( window, onresize ) ) ;

			// GLAD
			if ( ! gladLoadGLLoader( (GLADloadproc) glfwGetProcAddress ) ) {
				std::ostringstream comment ;
				
				comment << "GLAD error: gladLoadGLLoader failed" << std::endl ;
				throw std::runtime_error( comment.str() ) ;
			}

#ifdef __NVCC__
			// CUDA
			CUDA_CHECK( cudaSetDevice( 0 ) ) ; // 1st GPU assumed Turing

			int cdev, ddev ;
			// check if X server executes on current device (SO #17994896)
			// if true current is display device as required for GL interop
			CUDA_CHECK( cudaGetDevice( &cdev ) ) ;
			CUDA_CHECK( cudaDeviceGetAttribute( &ddev, cudaDevAttrKernelExecTimeout, cdev ) ) ;
			if ( ! ddev )
				throw std::runtime_error( "RTWO error: current not display device\n" ) ;
#endif // __NVCC__
		}



		// compile shaders
		GLuint v_shader = 0 ;
		GLuint f_shader = 0 ;
		{
			// vertex shader
			GL_CHECK( v_shader = glCreateShader( GL_VERTEX_SHADER ) ) ;
			const GLchar* src = reinterpret_cast<const GLchar*>( vert_glsl ) ;
			GL_CHECK( glShaderSource( v_shader, 1, &src, nullptr ) ) ;
			GL_CHECK( glCompileShader( v_shader ) ) ;
			GLint success = 0 ;
			GL_CHECK( glGetShaderiv( v_shader, GL_COMPILE_STATUS, &success ) ) ;
			if ( success == GL_FALSE ) {
				char log[512] ;
				glGetShaderInfoLog( v_shader, 512, nullptr, log ) ;
				glDeleteShader( v_shader ) ;
				std::ostringstream comment ;
				comment
					<< "vertex shader compilation failed: "
					<< log << std::endl ;
				throw std::runtime_error( comment.str() ) ;
			}
			// fragment shader
			GL_CHECK( f_shader = glCreateShader( GL_FRAGMENT_SHADER ) ) ;
			src = reinterpret_cast<const GLchar*>( frag_glsl ) ;
			GL_CHECK( glShaderSource( f_shader, 1, &src, nullptr ) ) ;
			GL_CHECK( glCompileShader( f_shader ) ) ;
			GL_CHECK( glGetShaderiv( f_shader, GL_COMPILE_STATUS, &success ) ) ;
			if ( success == GL_FALSE ) {
				char log[512] ;
				glGetShaderInfoLog( f_shader, 512, nullptr, log ) ;
				glDeleteShader( f_shader ) ;
				std::ostringstream comment ;
				comment
					<< "fragment shader compilation failed: "
					<< log << std::endl ;
				throw std::runtime_error( comment.str() ) ;
			}
		}



		// create program
		GLuint program = 0 ;
		GLint uniform = 0 ;
		{
			GL_CHECK( program = glCreateProgram() ) ;
			GL_CHECK( glAttachShader( program, v_shader ) ) ;
			GL_CHECK( glAttachShader( program, f_shader ) ) ;
			GL_CHECK( glLinkProgram( program ) ) ;
			GL_CHECK( glDetachShader( program, v_shader ) ) ;
			GL_CHECK( glDetachShader( program, f_shader ) ) ;
			GLint success = 0 ;
			GL_CHECK( glGetProgramiv( program, GL_LINK_STATUS, &success ) ) ;
			if ( success == GL_FALSE ) {
				char log[512] ;
				glGetProgramInfoLog( v_shader, 512, nullptr, log ) ;
				glDeleteProgram( program ) ;
				glDeleteShader( f_shader ) ;
				glDeleteShader( v_shader ) ;
				std::ostringstream comment ;
				comment
					<< "link shader program failed: "
					<< log << std::endl ;
				throw std::runtime_error( comment.str() ) ;
			}
			GL_CHECK( uniform = glGetUniformLocation( program, "t" ) ) ;
		}



		// setup vertex buffer object
		GLuint vao = 0 ;
		GLuint vbo = 0 ;
		{
			GL_CHECK( glGenVertexArrays( 1, &vao ) ) ;
			GL_CHECK( glBindVertexArray( vao ) ) ;
			// vertices of two triangles forming a rectangle and spanning the viewport (NDC)
			static const GLfloat vces[] = {
				-1.f, -1.f, .0f,
				 1.f, -1.f, .0f,
				-1.f,  1.f, .0f,

				-1.f,  1.f, .0f,
				 1.f, -1.f, .0f,
				 1.f,  1.f, .0f,
			} ;
			GL_CHECK( glGenBuffers( 1, &vbo ) ) ;
			GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, vbo ) ) ;
			GL_CHECK( glBufferData(
				GL_ARRAY_BUFFER,
				sizeof( vces ),
				vces,
				GL_STATIC_DRAW
				) ) ;
			GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) ) ;
		}



		// setup texture (image)
		GLuint tex = 0 ;
		GLuint pbo = 0 ;
#ifdef __NVCC__
		cudaGraphicsResource* glx = nullptr ;
#endif // __NVCC__
		{
			GL_CHECK( glGenTextures( 1, &tex ) ) ;
			GL_CHECK( glBindTexture( GL_TEXTURE_2D, tex ) ) ;
			GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) ) ;
			GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) ) ;
			GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) ) ;
			GL_CHECK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE ) ) ;
			// create pixel (image) buffer object
			GL_CHECK( glGenBuffers( 1, &pbo ) ) ;
			GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, pbo ) ) ;
#ifdef __NVCC__
			GL_CHECK( glBufferData(
				GL_ARRAY_BUFFER,
				c*w*h,
				nullptr,
				GL_STATIC_DRAW
				) ) ;
			// register pbo for CUDA
			CUDA_CHECK( cudaGraphicsGLRegisterBuffer( &glx, pbo, cudaGraphicsMapFlagsWriteDiscard ) ) ;
#else
			GL_CHECK( glBufferData(
				GL_ARRAY_BUFFER,
				c*w*h,
				image,
				GL_STATIC_DRAW
				) ) ;
#endif // __NVCC__
			GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) ) ;
		}



		// render loop
		do {
#ifdef __NVCC__
			CUstream cuda_stream ;
			CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;

			CUDA_CHECK( cudaGraphicsMapResources( 1, &glx, cuda_stream ) ) ;
			unsigned char* d_image ;
			size_t d_image_size ;
			CUDA_CHECK( cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void**>( &d_image ), &d_image_size, glx ) ) ;

			{
				CUDA_CHECK( cudaMemcpy(
					reinterpret_cast<void*>( d_image ),
					image,
					d_image_size, // c*w*h
					cudaMemcpyHostToDevice
					) ) ;
			}

			CUDA_CHECK( cudaGraphicsUnmapResources( 1, &glx, cuda_stream ) ) ;

			CUDA_CHECK( cudaStreamDestroy( cuda_stream ) ) ;
#endif // __NVCC__

			GL_CHECK( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) ) ;

			GL_CHECK( glClearColor( .333f, .525f, .643f, 1.f ) ) ;
			GL_CHECK( glClear( GL_COLOR_BUFFER_BIT ) ) ;

			GL_CHECK( glUseProgram( program ) ) ;
			GL_CHECK( glUniform1i( uniform , 0 ) ) ;

			GL_CHECK( glActiveTexture( GL_TEXTURE0 ) ) ;
			GL_CHECK( glBindTexture( GL_TEXTURE_2D, tex ) ) ;
			GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo ) ) ;
			GL_CHECK( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) ) ;
			GL_CHECK( glTexImage2D(
				GL_TEXTURE_2D,
				0,                // for mipmap level
				GL_RGB8,          // texture color components
				w,
				h,
				0,
				GL_RGB,           // pixel data format
				GL_UNSIGNED_BYTE, // pixel data type
				nullptr           // data in GL_PIXEL_UNPACK_BUFFER (pbo)
				) ) ;
			GL_CHECK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 ) ) ;

			GL_CHECK( glEnableVertexAttribArray( 0 ) ) ;
			GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, vbo ) ) ;
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

			GLFW_CHECK( glfwSwapBuffers( window ) ) ;
			GLFW_CHECK( glfwWaitEvents() ) ;
		} while ( ! glfwWindowShouldClose( window ) ) ;



		// cleanup
		{
#ifdef __NVCC__
			CUDA_CHECK( cudaGraphicsUnregisterResource( glx ) ) ;
#endif // __NVCC__
			GL_CHECK( glDeleteTextures( 1, &tex ) ) ;
			GL_CHECK( glDeleteBuffers( 1, &pbo ) ) ;
			GL_CHECK( glDeleteVertexArrays( 1, &vao ) ) ;
			GL_CHECK( glDeleteBuffers( 1, &vbo ) ) ;
			GL_CHECK( glDeleteShader( v_shader ) ) ;
			GL_CHECK( glDeleteShader( f_shader ) ) ;
			GL_CHECK( glDeleteProgram( program ) ) ;
			GLFW_CHECK( glfwDestroyWindow( window ) ) ;
			GLFW_CHECK( glfwTerminate() ) ;
			stbi_image_free( image ) ;
		}
	} catch ( std::exception& e ) {
		std::cerr << "exception: " << e.what() << std::endl ;

		return 1 ;
	}

	return( 0 ) ;
}
