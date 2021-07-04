#ifndef SIMPLEUI_H
#define SIMPLEUI_H

#include "camera.h"
#include "thing.h"
#include "rtwo.h"

class SimpleUI {
	public:
		SimpleUI( const std::string& name, LpGeneral& lp_general ) ;
		~SimpleUI() noexcept ( false ) ;

		void render( const OptixPipeline pipeline, const OptixShaderBindingTable& sbt ) ;

	private:
		LpGeneral lp_general_ ;
		GLFWwindow* window_ ;
		GLuint v_shader_ ;
		GLuint f_shader_ ;
		GLuint program_ ;
		GLint  uniform_ ;
		GLuint vao_ ;
		GLuint vbo_ ;
		GLuint tex_ ;
		GLuint pbo_ ;
		cudaGraphicsResource* glx_ ;
} ;

#endif // SIMPLEUI_H
