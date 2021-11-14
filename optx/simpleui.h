#ifndef SIMPLEUI_H
#define SIMPLEUI_H

// system includes
#include <string>

// subsystem includes
// OptiX
#include <optix.h>
// GLFW
#include <GLFW/glfw3.h>

// local includes
// none

// file specific includes
// none

class SimpleUI {
	public:
		SimpleUI( OptixDeviceContext& optx_context, const std::string& name ) ;
		~SimpleUI() noexcept ( false ) ;

		void render() ;

		static void usage() ;

	private:
		GLFWwindow* window_ ;
		GLuint v_shader_ ;
		GLuint f_shader_ ;
		GLuint program_ ;
		GLint  uniform_ ;
		GLuint vao_ ;
		GLuint vbo_ ;
		GLuint tex_ ;
} ;

#endif // SIMPLEUI_H
