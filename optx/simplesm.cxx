// system includes
// none

// subsystem includes
// OptiX
#include <optix.h>
#include <optix_stubs.h>
// GLAD
#include <glad/glad.h>
// GLFW
#include <GLFW/glfw3.h>
// CUDA
#include <cuda_gl_interop.h> // must follow glad.h

// local includes
#include "launcher.h"
#include "rtwo.h"
#include "v.h"

// file specific includes
#include "simplesm.h"

// common globals
namespace cg {
	extern Args*                   args ;
	extern LpGeneral               lp_general ;
	extern Launcher*               launcher ;
}
using namespace cg ;

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
		// reduce RT quality while animating
		SmFrame smframe = { lp_general.spp } ;
		h_values_.push( smframe ) ;
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
		SimpleSM::eaRdlDns() ; // RDL group action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::STL ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaStlEdt() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::EDT ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaAnmDns() {
	EA_ENTER() ;
	{ // perform action
		SimpleSM::eaRdlDns() ; // RDL group action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ANM ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaAnmEdt() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	// h_state_.pop() ;
	// h_event_.pop() ;
	// transition
	const State next = State::EDT ;
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
		// restore RT quality after animating
		SmFrame smframe = h_values_.top() ;
		lp_general.spp = smframe.spp ;
		h_values_.pop() ;
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
		SimpleSM::eaRdlDir() ; // RDL group action
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
		SimpleSM::eaRdlDir() ; // RDL group action
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
		SmFrame smframe = h_values_.top() ;
		lp_general.spp = smframe.spp ;
		h_values_.pop() ;
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
		SimpleSM::eaRdlPos() ; // RDL group action
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
		SimpleSM::eaRdlPos() ; // RDL group action
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
		SmFrame smframe = h_values_.top() ;
		lp_general.spp = smframe.spp ;
		h_values_.pop() ;
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
		paddle_->start( 0, 0 ) ;
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
		SimpleSM::eaRdlRsz() ; // RDL group action
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
		SimpleSM::eaRdlRsz() ; // RDL group action
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
		SmFrame smframe = { lp_general.spp } ;
		h_values_.push( smframe ) ;
		lp_general.spp = 1 ;
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
		SmFrame smframe = { lp_general.spp } ;
		h_values_.push( smframe ) ;
		lp_general.spp = 1 ;
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
		SmFrame smframe = h_values_.top() ;
		lp_general.spp = smframe.spp ;
		h_values_.pop() ;
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

void SimpleSM::eaEdtPos() {
	EA_ENTER() ;
	{ // perform action
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		lp_general.picker = true ;
		lp_general.pick_x =  static_cast<int>( x ) ;
		lp_general.pick_y = -static_cast<int>( y )+lp_general.image_h ;
		CUstream cuda_stream ;
		CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;
		launcher->ignite( cuda_stream, 1, 1 ) ;
		CUDA_CHECK( cudaStreamDestroy( cuda_stream ) ) ;
		lp_general.picker = false ;
		CUDA_CHECK( cudaMemcpy(
			reinterpret_cast<void*>( &smparam->pick_id ),
			lp_general.pick_id,
			sizeof( unsigned int ),
			cudaMemcpyDeviceToHost
			) ) ;
		std::cerr << "*** picked instance id " << smparam->pick_id << std::endl ;
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::OPO ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaEdtDir() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ODI ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaEdtRet() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = h_state_.top() ;
	h_state_.pop() ;
	h_event_.pop() ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaOpoRet() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::EDT ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaOdiRet() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::EDT ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaOdiMov() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ODI ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaOpoMov() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::OPO ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaOdiScr() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::ODI ;
	h_state_.push( next ) ;
	EA_LEAVE( next ) ;
}

void SimpleSM::eaOpoScr() {
	EA_ENTER() ;
	{ // perform action
	}
	// clear history (comment to keep)
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	const State next = State::OPO ;
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
	// select next denoiser type from list
	if ( ! smparam->denoiser )
		smparam->denoiser = new Denoiser( *smparam->optx_context, Dns::SMP ) ;
	else if ( smparam->denoiser->type() == Dns::SMP )
		smparam->denoiser = new Denoiser( *smparam->optx_context, Dns::NRM ) ;
	else if ( smparam->denoiser->type() == Dns::NRM )
		smparam->denoiser = new Denoiser( *smparam->optx_context, Dns::ALB ) ;
	else if ( smparam->denoiser->type() == Dns::ALB )
		smparam->denoiser = new Denoiser( *smparam->optx_context, Dns::NAA ) ;
	else if ( smparam->denoiser->type() == Dns::NAA )
		smparam->denoiser = new Denoiser( *smparam->optx_context, Dns::AOV ) ;
	else if ( smparam->denoiser->type() == Dns::AOV )
		smparam->denoiser = nullptr ;

	if ( smparam->denoiser && args->flag_v() )
		std::cerr << "denoiser " << static_cast<int>( smparam->denoiser->type() ) << std::endl ;
}

void SimpleSM::eaRdlDir() {
	double x, y ;
	glfwGetCursorPos( window_, &x, &y ) ;
	paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
	// reduce RT quality while moving
	SmFrame smframe = { lp_general.spp } ;
	h_values_.push( smframe ) ;
	lp_general.spp = 1 ;
}

void SimpleSM::eaRdlPos() {
	double x, y ;
	glfwGetCursorPos( window_, &x, &y ) ;
	paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
	// reduce RT quality while moving
	SmFrame smframe = { lp_general.spp } ;
	h_values_.push( smframe ) ;
	lp_general.spp = 1 ;
}

void SimpleSM::eaRdlRsz() {
	int w, h ;
	GLFW_CHECK( glfwGetFramebufferSize( window_, &w, &h ) ) ;
	SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
	lp_general.image_w = w ;
	lp_general.image_h = h ;
	Camera& camera = lp_general.camera ;
	camera.aspratio( static_cast<float>( w )/static_cast<float>( h ) ) ;
	// realloc device buffers
	launcher->resize( w, h ) ;
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
	if ( smparam->denoiser )
		smparam->denoiser = new Denoiser( *smparam->optx_context, smparam->denoiser->type() ) ;
}
