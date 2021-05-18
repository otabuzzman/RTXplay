#include <iostream>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define GLFWOK( api )                                      \
	if ( true ) {                                          \
		api ;                                              \
		const char* m ;                                    \
		if ( glfwGetError( &m ) != GLFW_NO_ERROR ) {       \
			std::ostringstream comment ;                   \
			comment << "GLFW error: " << m << std::endl ;  \
			throw std::runtime_error( comment.str() ) ;    \
		}                                                  \
	} else

#define GLOK( api )                                        \
	if ( true ) {                                          \
		api ;                                              \
		if ( glGetError() != GL_NO_ERROR ) {               \
			std::ostringstream comment ;                   \
			comment << "GL error: " << #api << std::endl ; \
			throw std::runtime_error( comment.str() ) ;    \
		}                                                  \
	} else

// GLSL sources of shaders
extern "C" const char vert_glsl[] ;
extern "C" const char frag_glsl[] ;

void onkey( GLFWwindow* window, const int keyname, const int keycode, const int keyact, const int keymod ) {
	glfwSetWindowShouldClose( window, GLFW_TRUE ) ;
}

void onresize( GLFWwindow* window, const int w, const int h ) {
	glViewport( 0, 0, w, h ) ;
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

	try {
		// initialize GLFW and GLAD
		GLFWwindow* window ;
		{
			GLFWOK( glfwInit() ) ;

			GLFWOK( glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 ) ) ;
			GLFWOK( glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 ) ) ;
			GLFWOK( glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE ) ) ;

			GLFWOK( window = glfwCreateWindow( w, h, "RTWO", nullptr, nullptr ) ) ;

			GLFWOK( glfwSetKeyCallback( window, onkey ) ) ;

			GLFWOK( glfwMakeContextCurrent( window ) ) ;

			if ( ! gladLoadGLLoader( (GLADloadproc) glfwGetProcAddress ) ) {
				std::ostringstream comment ;
				
				comment << "GLAD error: gladLoadGLLoader failed" << std::endl ;
				throw std::runtime_error( comment.str() ) ;
			}
		}



		// compile shaders
		GLuint v_shader = 0 ;
		GLuint f_shader = 0 ;
		{
			// vertex shader
			GLOK( v_shader = glCreateShader( GL_VERTEX_SHADER ) ) ;
			const GLchar* src = reinterpret_cast<const GLchar*>( vert_glsl ) ;
			GLOK( glShaderSource( v_shader, 1, &src, nullptr ) ) ;
			GLOK( glCompileShader( v_shader ) ) ;
			GLint success = 0 ;
			GLOK( glGetShaderiv( v_shader, GL_COMPILE_STATUS, &success ) ) ;
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
			GLOK( f_shader = glCreateShader( GL_FRAGMENT_SHADER ) ) ;
			src = reinterpret_cast<const GLchar*>( frag_glsl ) ;
			GLOK( glShaderSource( f_shader, 1, &src, nullptr ) ) ;
			GLOK( glCompileShader( f_shader ) ) ;
			GLOK( glGetShaderiv( f_shader, GL_COMPILE_STATUS, &success ) ) ;
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
		{
			GLOK( program = glCreateProgram() ) ;
			GLOK( glAttachShader( program, v_shader ) ) ;
			GLOK( glAttachShader( program, f_shader ) ) ;
			GLOK( glLinkProgram( program ) ) ;
			GLOK( glDetachShader( program, v_shader ) ) ;
			GLOK( glDetachShader( program, f_shader ) ) ;
			GLint success = 0 ;
			GLOK( glGetProgramiv( program, GL_LINK_STATUS, &success ) ) ;
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
		}



		// setup vertex buffer object
		GLuint vao = 0 ;
		GLuint vbo = 0 ;
		{
			GLOK( glGenVertexArrays( 1, &vao ) ) ;
			GLOK( glBindVertexArray( vao ) ) ;
			// square of two triangles spanning the viewport (NDC).
			static const GLfloat viewport[] = {
				-1.f, -1.f, .0f,
				 1.f, -1.f, .0f,
				-1.f,  1.f, .0f,

				-1.f,  1.f, .0f,
				 1.f, -1.f, .0f,
				 1.f,  1.f, .0f,
			} ;
			GLOK( glGenBuffers( 1, &vbo ) ) ;
			GLOK( glBindBuffer( GL_ARRAY_BUFFER, vbo ) ) ;
			GLOK( glBufferData(
				GL_ARRAY_BUFFER,
				sizeof( viewport ),
				viewport,
				GL_STATIC_DRAW
				) ) ;
		}



		// setup texture (image)
		GLuint tex = 0 ;
		GLuint ibo = 0 ;
		{
			GLOK( glGenTextures( 1, &tex ) ) ;
			GLOK( glBindTexture( GL_TEXTURE_2D, tex ) ) ;
			GLOK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ) ) ;
			GLOK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ) ) ;
			GLOK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) ) ;
			GLOK( glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE ) ) ;
			GLOK( glGenBuffers( 1, &ibo ) ) ;
			GLOK( glBindBuffer( GL_ARRAY_BUFFER, ibo ) ) ;
			GLOK( glBufferData(
				GL_ARRAY_BUFFER,
				c*w*h,
				image,
				GL_STATIC_DRAW
				) ) ;
		}



		// render loop
		do {
			/*** in case been set off-screen elsewhere 
			GLOK( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) ) ;
			***/
			GLOK( glViewport( 0, 0, w, h ) ) ;
			GLOK( glClear( GL_COLOR_BUFFER_BIT ) ) ;

			GLOK( glUseProgram( program ) ) ;

			GLOK( glActiveTexture( GL_TEXTURE0 ) ) ;
			GLOK( glBindTexture( GL_TEXTURE_2D, tex ) ) ;
			GLOK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, ibo ) ) ;
			GLOK( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) ) ;
			GLOK( glTexImage2D(
				GL_TEXTURE_2D,
				0,                // for mipmap level
				GL_RGB8,          // texture color components
				w,
				h,
				0,
				GL_RGB,           // pixel data format
				GL_UNSIGNED_BYTE, // pixel data type
				nullptr           // data in GL_PIXEL_UNPACK_BUFFER (ibo)
				) ) ;
			GLOK( glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 ) ) ;

			GLOK( glEnableVertexAttribArray( 0 ) ) ;
			GLOK( glBindBuffer( GL_ARRAY_BUFFER, vbo ) ) ;
			GLOK( glVertexAttribPointer(
				0,        // attribute index
				3,        // attribute components
				GL_FLOAT, // attribute components type
				GL_FALSE, // normalize fix-point
				0,        // stride between consecutive attributes
				nullptr   // data in GL_ARRAY_BUFFER (vbo)
				) ) ;

			/*** apply gamma correction
			GLOK( glEnable( GL_FRAMEBUFFER_SRGB ) ) ;
			***/
			GLOK( glDrawArrays( GL_TRIANGLES, 0, 6 ) ) ;
			GLOK( glDisableVertexAttribArray( 0 ) ) ;
			/***
			GLOK( glDisable( GL_FRAMEBUFFER_SRGB ) ) ;
			***/

			GLFWOK( glfwSwapBuffers( window ) ) ;
			GLFWOK( glfwPollEvents() ) ;
		} while ( ! glfwWindowShouldClose( window ) ) ;



		// cleanup
		{
			GLOK( glDeleteTextures( 1, &tex ) ) ;
			GLOK( glDeleteBuffers( 1, &ibo ) ) ;
			GLOK( glDeleteVertexArrays( 1, &vao ) ) ;
			GLOK( glDeleteBuffers( 1, &vbo ) ) ;
			GLOK( glDeleteShader( v_shader ) ) ;
			GLOK( glDeleteShader( f_shader ) ) ;
			GLOK( glDeleteProgram( program ) ) ;
			GLFWOK( glfwDestroyWindow( window ) ) ;
			GLFWOK( glfwTerminate() ) ;
			stbi_image_free( image ) ;
		}
	} catch ( std::exception& e ) {
		std::cerr << "exception: " << e.what() << std::endl ;

		return 1 ;
	}

	return( 0 ) ;
}
