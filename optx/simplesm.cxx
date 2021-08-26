#include <optix.h>
#include <optix_stubs.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h> // must follow glad.h

#include "v.h"

#include "simplesm.h"

SimpleSM::SimpleSM( GLFWwindow* window, const Args& args ) : window_( window ), args_( args ) {
	h_state_.push( State::STL ) ; // start state

	SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
	Camera* camera = &smparam->lp_general.camera ;
	paddle_ = new Paddle( camera->eye(), camera->pat(), camera->vup() ) ;
}

SimpleSM::~SimpleSM() {
	h_state_.pop() ;

	delete paddle_ ; paddle_ = nullptr ;
}

void SimpleSM::transition( const Event& event ) {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( event ) ;

	if ( args_.flag_tracesm() ) std::cerr << "SM transition " << eventName[e] << " " ;

	h_event_.push( event ) ;
	( this->*EATab[s][e] )() ;
}

void SimpleSM::eaStlAnm() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// toggle animation
		smparam->options ^= SM_OPTION_ANIMATE ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ANM ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaStlRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		glfwSetWindowShouldClose( window_, true ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::STL ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaStlDir() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// reduce RT quality while moving
		i_sexchg_.push( smparam->lp_general.spp ) ;
		i_sexchg_.push( smparam->lp_general.depth ) ;
		smparam->lp_general.spp = 1 ;
		smparam->lp_general.depth = 1 ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::DIR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaAnmStl() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// toggle animation
		smparam->options ^= SM_OPTION_ANIMATE ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::STL ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaAnmRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		glfwSetWindowShouldClose( window_, true ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ANM ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaAnmDir() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// reduce RT quality while moving
		i_sexchg_.push( smparam->lp_general.spp ) ;
		i_sexchg_.push( smparam->lp_general.depth ) ;
		smparam->lp_general.spp = 1 ;
		smparam->lp_general.depth = 1 ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::DIR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaDirScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// update camera
		Camera* camera = &smparam->lp_general.camera ;
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		const float len = V::len( camera->vup() ) ;
		camera->vup( len*paddle_->roll( static_cast<int>( y ) ) ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::DIR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaDirMov() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// update camera
		Camera* camera = &smparam->lp_general.camera ;
		const float3 eye = camera->eye() ;
		const float  len = V::len( eye-camera->pat() ) ;
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		camera->pat( eye-len*paddle_->move( static_cast<int>( x ), static_cast<int>( y ) ) ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::DIR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaDirRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// restore RT quality after moving
		smparam->lp_general.depth = i_sexchg_.top() ;
		i_sexchg_.pop() ;
		smparam->lp_general.spp = i_sexchg_.top() ;
		i_sexchg_.pop() ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition to history state and clear
	const State next = h_state_.top() ;
	h_state_.pop() ;
	h_event_.pop() ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaStlPos() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// reduce RT quality while moving
		i_sexchg_.push( smparam->lp_general.spp ) ;
		i_sexchg_.push( smparam->lp_general.depth ) ;
		smparam->lp_general.spp = 1 ;
		smparam->lp_general.depth = 1 ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::POS ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaAnmPos() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// reduce RT quality while moving
		i_sexchg_.push( smparam->lp_general.spp ) ;
		i_sexchg_.push( smparam->lp_general.depth ) ;
		smparam->lp_general.spp = 1 ;
		smparam->lp_general.depth = 1 ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::POS ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaPosMov() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// update camera
		Camera* camera = &smparam->lp_general.camera ;
		const float3 pat = camera->pat() ;
		const float  len = V::len( camera->eye()-pat ) ;
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		camera->eye( pat+len*paddle_->move( static_cast<int>( x ), static_cast<int>( y ) ) ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::POS ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaPosRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// restore RT quality after moving
		smparam->lp_general.depth = i_sexchg_.top() ;
		i_sexchg_.pop() ;
		smparam->lp_general.spp = i_sexchg_.top() ;
		i_sexchg_.pop() ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition to history state and clear
	const State next = h_state_.top() ;
	h_state_.pop() ;
	h_event_.pop() ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaStlRsz() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		int w, h ;
		GLFW_CHECK( glfwGetFramebufferSize( window_, &w, &h ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->lp_general.image_w = w ;
		smparam->lp_general.image_h = h ;
		Camera* camera = &smparam->lp_general.camera ;
		camera->aspratio( static_cast<float>( w )/static_cast<float>( h ) ) ;
		// realloc render buffer
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( smparam->lp_general.rawRGB ) ) ) ;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &smparam->lp_general.rawRGB ), sizeof( float3 )*w*h ) ) ;
		// realloc rays per pixel (rpp) buffer
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( smparam->lp_general.rpp ) ) ) ;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &smparam->lp_general.rpp ), sizeof( unsigned int )*w*h ) ) ;
		// resize pixel (image) buffer object
		GL_CHECK( glGenBuffers( 1, &smparam->pbo ) ) ;
		GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, smparam->pbo ) ) ;
		GL_CHECK( glBufferData(
			GL_ARRAY_BUFFER,
			sizeof( uchar4 )*w*h,
			nullptr,
			GL_STATIC_DRAW
			) ) ;
		// register pbo for CUDA
		CUDA_CHECK( cudaGraphicsGLRegisterBuffer( &smparam->glx, smparam->pbo, cudaGraphicsMapFlagsWriteDiscard ) ) ;
		GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::STL ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaStlZom() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// reduce RT quality while zooming
		i_sexchg_.push( smparam->lp_general.spp ) ;
		i_sexchg_.push( smparam->lp_general.depth ) ;
		smparam->lp_general.spp = 1 ;
		smparam->lp_general.depth = 1 ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::ZOM ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaStlBlr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::BLR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaStlFoc() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::FOC ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaAnmRsz() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		int w, h ;
		GLFW_CHECK( glfwGetFramebufferSize( window_, &w, &h ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->lp_general.image_w = w ;
		smparam->lp_general.image_h = h ;
		Camera* camera = &smparam->lp_general.camera ;
		camera->aspratio( static_cast<float>( w )/static_cast<float>( h ) ) ;
		// realloc render buffer
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( smparam->lp_general.rawRGB ) ) ) ;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &smparam->lp_general.rawRGB ), sizeof( float3 )*w*h ) ) ;
		// realloc rays per pixel (rpp) buffer
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( smparam->lp_general.rpp ) ) ) ;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &smparam->lp_general.rpp ), sizeof( unsigned int )*w*h ) ) ;
		// resize pixel (image) buffer object
		GL_CHECK( glGenBuffers( 1, &smparam->pbo ) ) ;
		GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, smparam->pbo ) ) ;
		GL_CHECK( glBufferData(
			GL_ARRAY_BUFFER,
			sizeof( uchar4 )*w*h,
			nullptr,
			GL_STATIC_DRAW
			) ) ;
		// register pbo for CUDA
		CUDA_CHECK( cudaGraphicsGLRegisterBuffer( &smparam->glx, smparam->pbo, cudaGraphicsMapFlagsWriteDiscard ) ) ;
		GL_CHECK( glBindBuffer( GL_ARRAY_BUFFER, 0 ) ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ANM ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaAnmZom() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// reduce RT quality while zooming
		i_sexchg_.push( smparam->lp_general.spp ) ;
		i_sexchg_.push( smparam->lp_general.depth ) ;
		smparam->lp_general.spp = 1 ;
		smparam->lp_general.depth = 1 ;
	}
	// clear history (comment to keep)
	// _state_.pop() ;
	// _event_.pop() ;
	// transition
	const State next = State::ZOM ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaAnmBlr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::BLR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaAnmFoc() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::FOC ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaZomScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// update camera
		Camera* camera = &smparam->lp_general.camera ;
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		const float adj = ( static_cast<float>( y )>0 ) ? 1.1f : 1/1.1f ;
		const float fov = camera->fov() ;
		camera->fov( adj*fov ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ZOM ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaZomRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// restore RT quality after zooming
		smparam->lp_general.depth = i_sexchg_.top() ;
		i_sexchg_.pop() ;
		smparam->lp_general.spp = i_sexchg_.top() ;
		i_sexchg_.pop() ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition to history state and clear
	const State next = h_state_.top() ;
	h_state_.pop() ;
	h_event_.pop() ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaBlrScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::BLR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaBlrRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition to history state and clear
	const State next = h_state_.top() ;
	h_state_.pop() ;
	h_event_.pop() ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaFocScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	const State next = State::FOC ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// update camera
		Camera* camera = &smparam->lp_general.camera ;
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		const float adj = ( static_cast<float>( y )>0 ) ? 1.25f : 1/1.25f ;
		const float apt = camera->aperture() ;
		camera->aperture( adj*apt ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaFocRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event in state " << stateName[s] << " ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition to history state and clear
	const State next = h_state_.top() ;
	h_state_.pop() ;
	h_event_.pop() ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[static_cast<int>( next )] << "new state now" << std::endl ;
}

void SimpleSM::eaReject() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;

	if ( args_.flag_tracesm() ) std::cerr << eventName[e] << " event rejected in state " << stateName[s] << std::endl ;
	h_event_.pop() ;
}
