#include <optix.h>
#include <optix_stubs.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h> // must follow glad.h

#include "v.h"

#include "simplesm.h"

// common globals
extern Args*     args ;
extern LpGeneral lp_general ;

SimpleSM::SimpleSM( GLFWwindow* window ) : window_( window ) {
	h_state_.push( State::CTL ) ; // start state

	Camera& camera = lp_general.camera ;
	paddle_ = new Paddle( camera.eye(), camera.pat(), camera.vup() ) ;
}

SimpleSM::~SimpleSM() {
	h_state_.pop() ;

	delete paddle_ ;

	SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
	smparam->denoiser = nullptr ;
}

void SimpleSM::transition( const Event& event ) {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( event ) ;

	if ( args->flag_tracesm() ) std::cerr << "SM event " << eventName[e] << " : " ;

	h_event_.push( event ) ;
	( this->*EATab[s][e] )() ;
}

void SimpleSM::eaCtlAnm() {
	EA_ENTER() ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// toggle animation
		smparam->anm_exec = ! smparam->anm_exec ;
		// toggle denoising
		smparam->dns_exec = ! smparam->dns_exec ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::CTL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaCtlDns() {
	EA_ENTER() ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->denoiser = nullptr ; // delete denoiser if set
		smparam->dns_type = ( smparam->dns_type+1 )%DNS_NUM ;
		switch ( smparam->dns_type ) {
			case DNS_SMP:
				smparam->denoiser = new DenoiserSMP( lp_general.image_w, lp_general.image_h ) ;
				break ;
			case DNS_NRM:
				break ;
			case DNS_ALB:
				break ;
			case DNS_NAA:
				break ;
			case DNS_AOV:
				break ;
			default: // DNS_NONE
				break ;
		}
		if ( args->flag_verbose() )
			std::cerr << "denoiser " << ( smparam->dns_exec ? smparam->dns_type : -1 ) << std::endl ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::CTL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaCtlRet() {
	EA_ENTER() ;
	{ // perform action
		glfwSetWindowShouldClose( window_, true ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::CTL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaCtlDir() {
	EA_ENTER() ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
		// reduce RT quality while moving
		i_sexchg_.push( lp_general.spp ) ;
		lp_general.spp = 1 ;
		// toggle denoising
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->dns_exec = ! smparam->dns_exec ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::DIR ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaDirScr() {
	EA_ENTER() ;
	{ // perform action
		// update camera
		Camera& camera = lp_general.camera ;
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		const float len = V::len( camera.vup() ) ;
		camera.vup( len*paddle_->roll( static_cast<int>( y ) ) ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::DIR ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaDirMov() {
	EA_ENTER() ;
	{ // perform action
		// update camera
		Camera& camera = lp_general.camera ;
		const float3 eye = camera.eye() ;
		const float  len = V::len( eye-camera.pat() ) ;
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		camera.pat( eye-len*paddle_->move( static_cast<int>( x ), static_cast<int>( y ) ) ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::DIR ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaDirRet() {
	EA_ENTER() ;
	{ // perform action
		// restore RT quality after moving
		lp_general.spp = i_sexchg_.top() ;
		i_sexchg_.pop() ;
		// toggle denoising
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->dns_exec = ! smparam->dns_exec ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition to history state and clear
	const State next = h_state_.top() ;
	h_state_.pop() ;
	h_event_.pop() ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaCtlPos() {
	EA_ENTER() ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
		// reduce RT quality while moving
		i_sexchg_.push( lp_general.spp ) ;
		lp_general.spp = 1 ;
		// toggle denoising
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->dns_exec = ! smparam->dns_exec ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::POS ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaPosMov() {
	EA_ENTER() ;
	{ // perform action
		// update camera
		Camera& camera = lp_general.camera ;
		const float3 pat = camera.pat() ;
		const float  len = V::len( camera.eye()-pat ) ;
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		camera.eye( pat+len*paddle_->move( static_cast<int>( x ), static_cast<int>( y ) ) ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::POS ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaPosRet() {
	EA_ENTER() ;
	{ // perform action
		// restore RT quality after moving
		lp_general.spp = i_sexchg_.top() ;
		i_sexchg_.pop() ;
		// toggle denoising
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->dns_exec = ! smparam->dns_exec ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition to history state and clear
	const State next = h_state_.top() ;
	h_state_.pop() ;
	h_event_.pop() ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaCtlRsz() {
	EA_ENTER() ;
	{ // perform action
		int w, h ;
		GLFW_CHECK( glfwGetFramebufferSize( window_, &w, &h ) ) ;
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		lp_general.image_w = w ;
		lp_general.image_h = h ;
		Camera& camera = lp_general.camera ;
		camera.aspratio( static_cast<float>( w )/static_cast<float>( h ) ) ;
		// realloc render buffer
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.rawRGB ) ) ) ;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.rawRGB ), sizeof( float3 )*w*h ) ) ;
		// realloc rays per pixel (rpp) buffer
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.rpp ) ) ) ;
		CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.rpp ), sizeof( unsigned int )*w*h ) ) ;
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
		// resize denoiser
		smparam->denoiser = nullptr ;
		switch ( smparam->dns_type ) {
			case DNS_SMP:
				smparam->denoiser = new DenoiserSMP( w, h ) ;
				break ;
			case DNS_NRM:
				break ;
			case DNS_ALB:
				break ;
			case DNS_NAA:
				break ;
			case DNS_AOV:
				break ;
			default: // DNS_NONE
				break ;
		}
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::CTL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaCtlZom() {
	EA_ENTER() ;
	{ // perform action
		// reduce RT quality while zooming
		i_sexchg_.push( lp_general.spp ) ;
		lp_general.spp = 1 ;
		// toggle denoising
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->dns_exec = ! smparam->dns_exec ;
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::ZOM ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaCtlBlr() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::BLR ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaCtlFoc() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::FOC ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaZomScr() {
	EA_ENTER() ;
	{ // perform action
		// update camera
		Camera& camera = lp_general.camera ;
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		const float adj = ( static_cast<float>( y )>0 ) ? 1.1f : 1/1.1f ;
		const float fov = camera.fov() ;
		camera.fov( adj*fov ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ZOM ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaZomRet() {
	EA_ENTER() ;
	{ // perform action
		// restore RT quality after zooming
		lp_general.spp = i_sexchg_.top() ;
		i_sexchg_.pop() ;
		// toggle denoising
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		smparam->dns_exec = ! smparam->dns_exec ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition to history state and clear
	const State next = h_state_.top() ;
	h_state_.pop() ;
	h_event_.pop() ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaBlrScr() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::BLR ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaBlrRet() {
	EA_ENTER() ;
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
	EA_LEAVE( next ) ;
}

void SimpleSM::eaFocScr() {
	EA_ENTER() ;
	const State next = State::FOC ;
	{ // perform action
		// update camera
		Camera& camera = lp_general.camera ;
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		const float adj = ( static_cast<float>( y )>0 ) ? 1.25f : 1/1.25f ;
		const float apt = camera.aperture() ;
		camera.aperture( adj*apt ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaFocRet() {
	EA_ENTER() ;
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
	EA_LEAVE( next ) ;
}

void SimpleSM::eaReject() {
	const int s = static_cast<int>( h_state_.top() ) ;

	if ( args->flag_tracesm() ) std::cerr << " rejected in state " << stateName[s] << std::endl ;
	h_event_.pop() ;
}
