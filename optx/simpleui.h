#ifndef SIMPLEUI_H
#define SIMPLEUI_H

#include "args.h"
#include "simplesm.h"

class SimpleUI {
	public:
		SimpleUI( const std::string& name, LpGeneral& lp_general, const Args& args ) ;
		~SimpleUI() noexcept ( false ) ;

		void render( const OptixPipeline pipeline, const OptixShaderBindingTable& sbt ) ;

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

		// command line arguments
		Args args_ ;
} ;

#endif // SIMPLEUI_H
