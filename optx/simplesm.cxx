// system includes
#include <cstring>

// subsystem includes
// none

// local includes
#include "launcher.h"
#include "rtwo.h"
#include "scene.h"
#include "v.h"

// file specific includes
#include "simplesm.h"

// common globals
namespace cg {
	extern Args*     args ;
	extern Scene*    scene ;
	extern LpGeneral lp_general ;
	extern Launcher* launcher ;
}
using namespace cg ;

template<unsigned int D>
static void matmul( float a[D*D] /* input/ result */, const float b[D*D] ) {
	float m[D*D] = { 0 } ;
	for ( unsigned int i = 0 ; D>i ; i++ )
		for ( unsigned int j = 0 ; D>j ; j++ )
			for ( unsigned int k = 0 ; D>k ; k++ )
				m[i*D+j] += a[i*D+k]*b[k*D+j] ;
	memcpy( &a[0], &m[0], sizeof( float )*D*D ) ;
}

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
		paddle_->reset( 0, 0 ) ;
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
		SimpleSM::eaEdtSed() ; // SED group action
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
		SimpleSM::eaEdtSed() ; // SED group action
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
		// recalibrate paddle
		Camera& camera = lp_general.camera ;
		paddle_->gauge( camera.eye(), camera.pat(), camera.vup() ) ;
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
		// restore RT quality after editing
		SmFrame smframe = h_values_.top() ;
		lp_general.spp = smframe.spp ;
		h_values_.pop() ;
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
		// restore RT quality after editing
		SmFrame smframe = h_values_.top() ;
		lp_general.spp = smframe.spp ;
		h_values_.pop() ;
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
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		float transform[12] ;
		scene->get( smparam->pick_id, &transform[0] ) ;
		// retrieve data from instance transform
		const float3 pat = { transform[0*4+3], transform[1*4+3], transform[2*4+3] } ; // thing's center position vector
		float msr[9] = {
			transform[0*4+0], transform[0*4+1], transform[0*4+2], // thing's SR matrix
			transform[1*4+0], transform[1*4+1], transform[1*4+2],
			transform[2*4+0], transform[2*4+1], transform[2*4+2],
		} ;
		// set up x/ y rotate matrices
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		const float3 vrx = paddle_->move( 0, static_cast<int>( y ) ) ;          // paddle vector of x-axis rotation
		const float cosx = V::dot( pat, vrx )/( V::len( pat )*V::len( vrx ) ) ; // paddle angle
		const float sinx = sqrtf( 1.f-cosx*cosx ) ;
		const float rox[9] = {
			1.f,  0.f,   0.f,
			0.f, cosx, -sinx,
			0.f, sinx,  cosx
		} ;
		const float3 vry = paddle_->move( static_cast<int>( x ), 0 ) ;          // paddle vector of y-axis rotation
		const float cosy = V::dot( vrx, vry )/( V::len( vrx )*V::len( vry ) ) ;
		const float siny = sqrtf( 1.f-cosy*cosy ) ;
		const float roy[9] = {
			 cosy, 0.f, siny,
			  0.f, 1.f, 0.f,
			-siny, 0.f, cosy
		} ;
		// combine SR with x/ y rotate matrices
		matmul<3>( &msr[0], &rox[0] ) ;
		matmul<3>( &msr[0], &roy[0] ) ;
		// update instance transform
		transform[0*4+0] = msr[0*3+0] ; transform[0*4+1] = msr[0*3+1] ; transform[0*4+2] = msr[0*3+2] ;
		transform[1*4+0] = msr[1*3+0] ; transform[1*4+1] = msr[1*3+1] ; transform[1*4+2] = msr[1*3+2] ;
		transform[2*4+0] = msr[2*3+0] ; transform[2*4+1] = msr[2*3+1] ; transform[2*4+2] = msr[2*3+2] ;
		scene->set( smparam->pick_id, &transform[0] ) ;
		scene->update( lp_general.is_handle ) ;
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
		SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
		float transform[12] ;
		scene->get( smparam->pick_id, &transform[0] ) ;
		float3 pat = { transform[0*4+3], transform[1*4+3], transform[2*4+3] } ;
		Camera& camera = lp_general.camera ;
		const float3 eye = camera.eye() ;
		const float  len = V::len( eye-pat ) ;
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		pat = eye-len*paddle_->move( static_cast<int>( x ), static_cast<int>( y ) ) ;
		transform[0*4+3] = pat.x ;
		transform[1*4+3] = pat.y ;
		transform[2*4+3] = pat.z ;
		scene->set( smparam->pick_id, &transform[0] ) ;
		scene->update( lp_general.is_handle ) ;
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
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		// paddle_->move( static_cast<int>( x ), static_cast<int>( y ) ) ) ;
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
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		// paddle_->move( static_cast<int>( x ), static_cast<int>( y ) ) ) ;
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
	else if ( smparam->denoiser->type() == Dns::SMP ) {
		delete smparam->denoiser ;
		smparam->denoiser = new Denoiser( *smparam->optx_context, Dns::NRM ) ;
	} else if ( smparam->denoiser->type() == Dns::NRM ) {
		delete smparam->denoiser ;
		smparam->denoiser = new Denoiser( *smparam->optx_context, Dns::ALB ) ;
	} else if ( smparam->denoiser->type() == Dns::ALB ) {
		delete smparam->denoiser ;
		smparam->denoiser = new Denoiser( *smparam->optx_context, Dns::NAA ) ;
	} else if ( smparam->denoiser->type() == Dns::NAA ) {
		delete smparam->denoiser ;
		smparam->denoiser = new Denoiser( *smparam->optx_context, Dns::AOV ) ;
	} else if ( smparam->denoiser->type() == Dns::AOV ) {
		delete smparam->denoiser ;
		smparam->denoiser = nullptr ;
	}

	if ( smparam->denoiser && args->flag_v() )
		std::cerr << "denoiser " << static_cast<int>( smparam->denoiser->type() ) << std::endl ;
}

void SimpleSM::eaRdlDir() {
	double x, y ;
	glfwGetCursorPos( window_, &x, &y ) ;
	paddle_->reset( static_cast<int>( x ), static_cast<int>( y ) ) ;
	// reduce RT quality while moving
	SmFrame smframe = { lp_general.spp } ;
	h_values_.push( smframe ) ;
	lp_general.spp = 1 ;
}

void SimpleSM::eaRdlPos() {
	double x, y ;
	glfwGetCursorPos( window_, &x, &y ) ;
	paddle_->reset( static_cast<int>( x ), static_cast<int>( y ) ) ;
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

void SimpleSM::eaEdtSed() {
	SmParam* smparam = static_cast<SmParam*>( glfwGetWindowUserPointer( window_ ) ) ;
	double x, y ;
	glfwGetCursorPos( window_, &x, &y ) ;
	paddle_->reset( static_cast<int>( x ), static_cast<int>( y ) ) ;
	// one shot thru selected pixel
	lp_general.picker = true ;
	lp_general.pick_x =  static_cast<int>( x ) ;
	lp_general.pick_y = -static_cast<int>( y )+lp_general.image_h ;
	CUstream cuda_stream ;
	CUDA_CHECK( cudaStreamCreate( &cuda_stream ) ) ;
	launcher->ignite( cuda_stream, true ) ;
	CUDA_CHECK( cudaStreamDestroy( cuda_stream ) ) ;
	lp_general.picker = false ;
	// retrieve picked instance id
	CUDA_CHECK( cudaMemcpy(
		reinterpret_cast<void*>( &smparam->pick_id ),
		lp_general.pick_id,
		sizeof( unsigned int ),
		cudaMemcpyDeviceToHost
		) ) ;
	// recalibrate paddle
	float transform[12] ;
	scene->get( smparam->pick_id, &transform[0] ) ;
	const float3 pat = { transform[0*4+3], transform[1*4+3], transform[2*4+3] } ;
	Camera& camera = lp_general.camera ;
	paddle_->gauge( camera.eye(), pat, camera.vup() ) ;
	// reduce RT quality while editing
	SmFrame smframe = { lp_general.spp } ;
	h_values_.push( smframe ) ;
	lp_general.spp = 1 ;

	if ( args->flag_v() )
		std::cerr << "picked instance id " << smparam->pick_id << std::endl ;
}
