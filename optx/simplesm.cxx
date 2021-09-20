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
	h_state_.push( State::STL ) ; // start state

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

	if ( args->flag_t() ) std::cerr << "SM event " << event_name[e] << " : " ;

	h_event_.push( event ) ;
	( this->*EATab[s][e] )() ;
}

void SimpleSM::eaStlAnm() {
	EA_ENTER() ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// set render loop to poll events
		smparam->glfwPoWaEvents = &glfwPollEvents ;
		// turn on denoiser
		smparam->dns_exec = true ;
		// reduce RT quality while animating
		i_sexchg_.push( lp_general.spp ) ;
		lp_general.spp = 1 ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ANM ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaStlDns() {
	EA_ENTER() ;
	{ // perform action
		eaRdlDns() ; // RDL group action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::STL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaAnmDns() {
	EA_ENTER() ;
	{ // perform action
		eaRdlDns() ; // RDL group action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ANM ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaStlRet() {
	EA_ENTER() ;
	{ // perform action
		glfwSetWindowShouldClose( window_, true ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::STL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaAnmRet() {
	EA_ENTER() ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		// set render loop to wait for events
		smparam->glfwPoWaEvents = &glfwWaitEvents ;
		// turn off denoiser
		smparam->dns_exec = false ;
		// restore RT quality after animating
		lp_general.spp = i_sexchg_.top() ;
		i_sexchg_.pop() ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::STL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaStlDir() {
	EA_ENTER() ;
	{ // perform action
		eaRdlDir() ; // RDL group action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::DIR ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaAnmDir() {
	EA_ENTER() ;
	{ // perform action
		eaRdlDir() ; // RDL group action
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

void SimpleSM::eaStlPos() {
	EA_ENTER() ;
	{ // perform action
		eaRdlPos() ; // RDL group action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::POS ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaAnmPos() {
	EA_ENTER() ;
	{ // perform action
		eaRdlPos() ; // RDL group action
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

void SimpleSM::eaStlPcd() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::STL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaAnmPcd() {
	EA_ENTER() ;
	{ // perform action
		// update camera
		Camera& camera = lp_general.camera ;
		const float3 eye = camera.eye() ;
		const float3 pat = camera.pat() ;
		const float  len = V::len( eye-pat ) ;
		camera.eye( pat+len*paddle_->move( -1, 0 ) ) ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ANM ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaStlRsz() {
	EA_ENTER() ;
	{ // perform action
		eaRdlRsz() ; // RDL group action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::STL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaAnmRsz() {
	EA_ENTER() ;
	{ // perform action
		eaRdlRsz() ; // RDL group action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ANM ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaStlZom() {
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

void SimpleSM::eaAnmZom() {
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

void SimpleSM::eaStlBlr() {
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

void SimpleSM::eaAnmBlr() {
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

void SimpleSM::eaStlFoc() {
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

void SimpleSM::eaAnmFoc() {
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

	if ( args->flag_t() ) std::cerr << " rejected in state " << state_name[s] << std::endl ;
	h_event_.pop() ;
}

void SimpleSM::eaRdlDns() {
	SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
	smparam->denoiser = nullptr ; // delete denoiser
	if ( lp_general.normals ) {
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.normals ) ) ) ;
		lp_general.normals = nullptr ;
	}
	if ( lp_general.albedos ) {
		CUDA_CHECK( cudaFree( reinterpret_cast<void*>( lp_general.albedos ) ) ) ;
		lp_general.albedos = nullptr ;
	}
	// select next denoiser type from list
	if ( smparam->dns_type == Dns::AOV ) {
		smparam->dns_type = Dns::NONE ;
	} else {
		const int w = lp_general.image_w ;
		const int h = lp_general.image_h ;
		if ( smparam->dns_type == Dns::NONE )
			smparam->dns_type = Dns::SMP ;
		else if ( smparam->dns_type == Dns::SMP ) {
			smparam->dns_type = Dns::NRM ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.normals ), sizeof( float3 )*w*h ) ) ;
		} else if ( smparam->dns_type == Dns::NRM ) {
			smparam->dns_type = Dns::ALB ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.albedos ), sizeof( float3 )*w*h ) ) ;
		} else if ( smparam->dns_type == Dns::ALB ) {
			smparam->dns_type = Dns::NAA ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.normals ), sizeof( float3 )*w*h ) ) ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.albedos ), sizeof( float3 )*w*h ) ) ;
		} else { // smparam->dns_type == Dns::NAA
			smparam->dns_type = Dns::AOV ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.normals ), sizeof( float3 )*w*h ) ) ;
			CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &lp_general.albedos ), sizeof( float3 )*w*h ) ) ;
		}
		smparam->denoiser = new Denoiser( smparam->dns_type, w, h ) ;
	}
	if ( args->flag_v() ) {
		const char *mnemonic ;
		args->param_D( smparam->dns_type, &mnemonic ) ;
		std::cerr << "denoiser " << mnemonic << std::endl ;
	}
}

void SimpleSM::eaRdlDir() {
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

void SimpleSM::eaRdlPos() {
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

void SimpleSM::eaRdlRsz() {
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
	smparam->denoiser = nullptr ; // delete denoiser
	if ( smparam->dns_type != Dns::NONE )
		smparam->denoiser = new Denoiser( smparam->dns_type, lp_general.image_w, lp_general.image_h ) ;
}
