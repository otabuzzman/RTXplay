#ifndef SIMPLESM_H
#define SIMPLESM_H

#include <stack>

#include "args.h"
#include "camera.h"
#include "denoiser.h"
#include "thing.h"
#include "rtwo.h"
#include "paddle.h"

// missing in GLFW
extern "C" void glfwGetScroll( GLFWwindow* /*window*/, double* xscroll, double* yscroll ) ;

enum class State { BLR, CTL, DIR, FOC, POS, ZOM, n } ;
static const std::string stateName[] = { "BLR", "CTL", "DIR", "FOC", "POS", "ZOM" } ;
enum class Event { ANM, BLR, DIR, DNS, FOC, MOV, POS, RET, RSZ, SCR, ZOM, n } ;
static const std::string eventName[] = { "ANM", "BLR", "DIR", "DNS", "FOC", "MOV", "POS", "RET", "RSZ", "SCR", "ZOM" } ;

struct SmParam {
	GLuint                pbo ;
	cudaGraphicsResource* glx ;

	bool                  anm_exec = false ;

	bool                  dns_exec = false ;
	int                   dns_type = DNS_NONE ;
	Denoiser*             denoiser = nullptr ;
} ;

#define EA_ENTER()                                                                           \
	if ( args->flag_tracesm() ) {                                                            \
		const int s = static_cast<int>( h_state_.top() ) ;                                   \
		std::cerr << " changing state " << stateName[s] << " ... " ;                         \
	} else

#define EA_LEAVE( state )                                                                    \
	if ( args->flag_tracesm() ) {                                                            \
		std::cerr << "new state now " << stateName[static_cast<int>( state )] << std::endl ; \
	} else

class SimpleSM {
	public:
		SimpleSM( GLFWwindow* window ) ;
		~SimpleSM() ;

		void transition( const Event& event ) ;

		// event/ action implementations
		void eaCtlAnm() ;
		void eaCtlDns() ;
		void eaCtlRet() ;
		void eaCtlDir() ;
		void eaAnmCtl() ;
		void eaDirScr() ;
		void eaDirMov() ;
		void eaDirRet() ;
		void eaCtlPos() ;
		void eaPosMov() ;
		void eaPosRet() ;
		void eaCtlRsz() ;
		void eaCtlZom() ;
		void eaCtlBlr() ;
		void eaCtlFoc() ;
		void eaZomScr() ;
		void eaZomRet() ;
		void eaBlrScr() ;
		void eaBlrRet() ;
		void eaFocScr() ;
		void eaFocRet() ;

	private:
		GLFWwindow* window_ ;
		Paddle* paddle_ ;

		// state/ event history
		std::stack<State> h_state_ ;
		std::stack<Event> h_event_ ;

		// inter-state exchange
		std::stack<int>   i_sexchg_ ;
		// std::stack<float> f_sexchg_ ;
		// std::stack<void*> p_sexchg_ ;

		// event/ action table
		typedef void ( SimpleSM::*EAImp )() ;
		EAImp EATab[static_cast<int>( State::n )][static_cast<int>( Event::n )] = {
			/* S/ E ANM                  BLR                  DIR                  DNS                  FOC                  MOV                  POS                  RET                  RSZ                  SCR                  ZOM               */
			/*BLR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaBlrRet, &SimpleSM::eaReject, &SimpleSM::eaBlrScr, &SimpleSM::eaReject,
			/*CTL*/ &SimpleSM::eaCtlAnm, &SimpleSM::eaCtlBlr, &SimpleSM::eaCtlDir, &SimpleSM::eaCtlDns, &SimpleSM::eaCtlFoc, &SimpleSM::eaReject, &SimpleSM::eaCtlPos, &SimpleSM::eaCtlRet, &SimpleSM::eaCtlRsz, &SimpleSM::eaReject, &SimpleSM::eaCtlZom,
			/*DIR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaDirMov, &SimpleSM::eaReject, &SimpleSM::eaDirRet, &SimpleSM::eaReject, &SimpleSM::eaDirScr, &SimpleSM::eaReject,
			/*FOC*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaFocRet, &SimpleSM::eaReject, &SimpleSM::eaFocScr, &SimpleSM::eaReject,
			/*POS*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaPosMov, &SimpleSM::eaReject, &SimpleSM::eaPosRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*ZOM*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaZomRet, &SimpleSM::eaReject, &SimpleSM::eaZomScr, &SimpleSM::eaReject
		} ;

		void eaReject() ;
} ;

#endif // SIMPLESM_H
