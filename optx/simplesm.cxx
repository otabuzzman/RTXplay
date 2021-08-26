#include <optix.h>
#include <optix_stubs.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h> // must follow glad.h

#include "v.h"

#include "simplesm.h"

SimpleSM::SimpleSM( GLFWwindow* window, const Args& args ) : window_( window ), args_( args ) {
	h_state_.push( State::CTL ) ; // start state

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

	if ( args_.flag_tracesm() ) std::cerr << "SM transition: " ;

	h_event_.push( event ) ;
	( this->*EATab[s][e] )() ;
}

void SimpleSM::eaCtlAnm() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// toggle animation
		smparam->options ^= SM_OPTION_ANIMATE ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::CTL ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaCtlRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
		glfwSetWindowShouldClose( window_, true ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::CTL ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaCtlDir() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// reduce RT quality while moving
		i_sexchg_.push( smparam->lp_general.spp ) ;
		smparam->lp_general.spp = 1 ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::DIR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaDirScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaDirMov() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaDirRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// restore RT quality after moving
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaCtlPos() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// reduce RT quality while moving
		i_sexchg_.push( smparam->lp_general.spp ) ;
		smparam->lp_general.spp = 1 ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::POS ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaPosMov() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaPosRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// restore RT quality after moving
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaCtlRsz() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
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
	const State next = State::CTL ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaCtlZom() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// reduce RT quality while zooming
		i_sexchg_.push( smparam->lp_general.spp ) ;
		smparam->lp_general.spp = 1 ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::ZOM ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaCtlBlr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::BLR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaCtlFoc() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::FOC ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaZomScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaZomRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// restore RT quality after zooming
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaBlrScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::BLR ;
	h_state_.push( next ) ;
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaBlrRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaFocScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaFocRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;
	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state received " << eventName[e] << " event ... " ;
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
	if ( args_.flag_tracesm() ) std::cerr << "new state now " << stateName[static_cast<int>( next )] << std::endl ;
}

void SimpleSM::eaReject() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( h_event_.top() ) ;

	if ( args_.flag_tracesm() ) std::cerr << stateName[s] << " state rejected " << eventName[e] << " event" << std::endl ;
	h_event_.pop() ;
}
