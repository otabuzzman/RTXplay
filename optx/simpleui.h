#ifndef SIMPLEUI_H
#define SIMPLEUI_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h> // must follow glad.h

#include "camera.h"
#include "thing.h"
#include "rtwo.h"

class SimpleUI {
	public:
		SimpleUI( LpGeneral& lp_general ) ;
		~SimpleUI() noexcept ( false ) ;

		void render( const OptixPipeline pipeline, const OptixShaderBindingTable& sbt ) ;

	private:
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
		CUdeviceptr d_lp_general ;
} ;

#endif // SIMPLEUI_H
