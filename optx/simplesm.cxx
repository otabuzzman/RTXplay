#include <optix.h>
#include <optix_stubs.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h> // must follow glad.h

#include "v.h"

#include "simplesm.h"

SimpleSM::SimpleSM( GLFWwindow* window ) : window_( window ) {
	h_state_.push( State::CTL ) ;

	SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
	Camera* camera = &smparam->lp_general.camera ;
	paddle_ = new Paddle( camera->eye()-camera->pat() ) ;
}

SimpleSM::~SimpleSM() {
	delete paddle_ ; paddle_ = nullptr ;
}

void SimpleSM::transition( const Event& event ) {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( event ) ;

	std::cerr << "SM transition " << eventName[e] << " " ;

	h_event_.push( event ) ;
	( this->*EATab[s][e] )() ;
}

void SimpleSM::eaCtlRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
		glfwSetWindowShouldClose( window_, true ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaCtlDir() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::DIR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
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
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaDirScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::DIR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaDirMov() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::DIR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->track( static_cast<int>( x ), static_cast<int>( y ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// update camera
		Camera* camera = &smparam->lp_general.camera ;
		const float3 eye = camera->eye() ;
		const float  len = V::len( eye-camera->pat() ) ;
		camera->pat( eye-len*paddle_->hand() ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaDirRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// restore RT quality after moving
		smparam->lp_general.depth = i_sexchg_.top() ;
		i_sexchg_.pop() ;
		smparam->lp_general.spp = i_sexchg_.top() ;
		i_sexchg_.pop() ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaCtlPos() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::POS ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
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
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaPosMov() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::POS ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->track( static_cast<int>( x ), static_cast<int>( y ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// update camera
		Camera* camera = &smparam->lp_general.camera ;
		const float3 pat = camera->pat() ;
		const float  len = V::len( camera->eye()-pat ) ;
		camera->eye( pat+len*paddle_->hand() ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaPosRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// restore RT quality after moving
		smparam->lp_general.depth = i_sexchg_.top() ;
		i_sexchg_.pop() ;
		smparam->lp_general.spp = i_sexchg_.top() ;
		i_sexchg_.pop() ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaCtlRsz() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
		int w, h ;
		GLFW_CHECK( glfwGetFramebufferSize( window_, &w, &h ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->lp_general.image_w = w ;
		smparam->lp_general.image_h = h ;
		Camera* camera = &smparam->lp_general.camera ;
		camera->aspratio( static_cast<float>( w )/static_cast<float>( h ) ) ;
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
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaCtlZom() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::ZOM ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaCtlBlr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::BLR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaCtlFoc() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::FOC ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaZomScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::ZOM ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// update camera
		Camera* camera = &smparam->lp_general.camera ;
		const float adj = ( static_cast<float>( y )>0 ) ? 1.1f : 1/1.1f ;
		const float fov = camera->fov() ;
		camera->fov( adj*fov ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaZomRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaBlrScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::BLR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaBlrRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaFocScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::FOC ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// update camera
		Camera* camera = &smparam->lp_general.camera ;
		const float adj = ( static_cast<float>( y )>0 ) ? 1.25f : 1/1.25f ;
		const float apt = camera->aperture() ;
		camera->aperture( adj*apt ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaFocRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] << " ... " ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << "finished" << std::endl ;
}

void SimpleSM::eaReject() {
	const int s = static_cast<int>( h_state_.top() ) ;

	std::cerr << "rejected in " << stateName[s] << std::endl ;
	h_event_.pop() ;
}
