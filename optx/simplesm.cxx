#include <optix.h>
#include <optix_stubs.h>

#include <GLFW/glfw3.h>

#include "v.h"

#include "simplesm.h"

SimpleSM::SimpleSM( GLFWwindow* window, LpGeneral* lp_general ) : window_( window ), lp_general_( lp_general ) {
	h_state_.push( State::CTL ) ;

	paddle_ = new Paddle( V::unitV( lp_general_->camera.eye()-lp_general_->camera.pat() ) ) ;
}

SimpleSM::~SimpleSM() {
	delete paddle_ ; paddle_ = nullptr ;
}

void SimpleSM::transition( const Event& event ) {
	const int s = static_cast<int>( h_state_.top() ) ;
	const int e = static_cast<int>( event ) ;

	std::cerr << "SM transition " << eventName[e] << " ... " ;

	h_event_.push( event ) ;
	( this->*EATab[s][e] )() ;
}

void SimpleSM::eaCtlRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
		glfwSetWindowShouldClose( window_, true ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaCtlDir() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::DIR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaDirScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::DIR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaDirMov() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::DIR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->track( static_cast<int>( x ), static_cast<int>( y ) ) ;
		// update camera
		const float3 eye = lp_general_->camera.eye() ;
		const float  len = V::len( eye-lp_general_->camera.pat() ) ;
		lp_general_->camera.pat( eye-len*paddle_->hand() ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaDirRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaCtlPos() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::POS ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->start( static_cast<int>( x ), static_cast<int>( y ) ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaPosMov() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::POS ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
		double x, y ;
		glfwGetCursorPos( window_, &x, &y ) ;
		paddle_->track( static_cast<int>( x ), static_cast<int>( y ) ) ;
		// update camera
		const float3 pat = lp_general_->camera.pat() ;
		const float  len = V::len( lp_general_->camera.eye()-pat ) ;
		lp_general_->camera.eye( pat+len*paddle_->hand() ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaPosRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaCtlRsz() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaCtlZom() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::ZOM ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaCtlBlr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::BLR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaCtlFoc() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::FOC ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaZomScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::ZOM ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		// update camera
		const float fov = lp_general_->camera.fov() ;
		const float adj = ( static_cast<float>( y )>0 ) ? 1.1f : 1/1.1f ;
		lp_general_->camera.fov( adj*fov ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaZomRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaBlrScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::BLR ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaBlrRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaFocScr() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::FOC ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
		double x, y ;
		glfwGetScroll( window_, &x, &y ) ;
		// update camera
		const float apt = lp_general_->camera.aperture() ;
		const float adj = ( static_cast<float>( y )>0 ) ? 1.1f : 1/1.1f ;
		lp_general_->camera.aperture( adj*apt ) ;
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaFocRet() {
	const int s = static_cast<int>( h_state_.top() ) ;
	const State next = State::CTL ;
	std::cerr << "from " << stateName[s] << " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaReject() {
	const int s = static_cast<int>( h_state_.top() ) ;

	std::cerr << "rejected in " << stateName[s] << std::endl ;
	h_event_.pop() ;
}
