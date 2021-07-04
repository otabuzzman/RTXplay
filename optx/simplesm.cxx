#include <optix.h>
#include <optix_stubs.h>

#include <GLFW/glfw3.h>

#include "v.h"

#include "simplesm.h"

SimpleSM::SimpleSM( GLFWwindow* window, LpGeneral* lp_general ) : window_( window ), lp_general_( lp_general ) {
	h_state_.push( State::CTL ) ;

	paddle.set( V::unitV( lp_general_->camera.eye()-lp_general_->camera.pat() ) ) ;
}

void SimpleSM::transition( const Event event ) {
	std::cerr << "SM transition " << eventName[static_cast<int>( event )] << " ... " ;

	h_event_.push( event ) ;
	( this->*EATab[static_cast<int>( h_state_.top() )][static_cast<int>( event )] )() ;
}

void SimpleSM::eaCtlRet() {
	const State next = State::CTL ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaCtlDir() {
	const State next = State::DIR ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaDirScr() {
	const State next = State::DIR ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::DIR ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaDirRet() {
	const State next = State::CTL ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::POS ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaPosMov() {
	const State next = State::POS ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaPosRet() {
	const State next = State::CTL ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::CTL ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::ZOM ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::BLR ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::FOC ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::ZOM ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaZomRet() {
	const State next = State::CTL ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::BLR ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::CTL ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	const State next = State::FOC ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
	{ // perform action
	}
	// set history
	h_state_.pop() ;
	h_event_.pop() ;
	// transition
	h_state_.push( next ) ;
	std::cerr << " finished" << std::endl ;
}

void SimpleSM::eaFocRet() {
	const State next = State::CTL ;
	std::cerr
		<< "from " << stateName[static_cast<int>( h_state_.top() )]
		<< " to "  << stateName[static_cast<int>( next )] ;
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
	std::cerr << "rejected in " << stateName[static_cast<int>( h_state_.top() )] << std::endl ;
	h_event_.pop() ;
}
