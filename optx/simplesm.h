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

enum class State { ANM, BLR, DIR, FOC, POS, STL, ZOM, n } ;
static const std::string state_name[] = { "ANM", "BLR", "DIR", "FOC", "POS", "STL", "ZOM" } ;
enum class Event { ANM, BLR, DIR, DNS, FOC, MOV, PCD, POS, RET, RSZ, SCR, ZOM, n } ;
static const std::string event_name[] = { "ANM", "BLR", "DIR", "DNS", "FOC", "MOV", "PCD", "POS", "RET", "RSZ", "SCR", "ZOM" } ;

struct SmParam {
	GLuint                pbo ;
	cudaGraphicsResource* glx ;

	void                  ( *glfwPoWaEvents )() = &glfwWaitEvents ;

	bool                  dns_exec              = false ;
	Dns                   dns_type              = Dns::NONE ;
	Denoiser*             denoiser              = nullptr ;
} ;

// preserve state history values
union SmFrame {
	unsigned int          spp ;
} ;

#define EA_ENTER()                                                                                \
	do {                                                                                          \
		if ( args->flag_t() ) {                                                                   \
			const int s = static_cast<int>( h_state_.top() ) ;                                    \
			std::cerr << " changing state " << state_name[s] << " ... " ;                         \
		}                                                                                         \
	} while ( false )

#define EA_LEAVE( state )                                                                         \
	do {                                                                                          \
		if ( args->flag_t() ) {                                                                   \
			std::cerr << "new state now " << state_name[static_cast<int>( state )] << std::endl ; \
		}                                                                                         \
	} while ( false )

class SimpleSM {
	public:
		SimpleSM( GLFWwindow* window ) ;
		~SimpleSM() ;

		void transition( const Event& event ) ;

		// event/ action implementations
		void eaStlAnm() ;
		void eaStlDns() ;
		void eaAnmDns() ;
		void eaStlRet() ;
		void eaAnmRet() ;
		void eaStlDir() ;
		void eaAnmDir() ;
		void eaDirScr() ;
		void eaDirMov() ;
		void eaDirRet() ;
		void eaStlPos() ;
		void eaAnmPos() ;
		void eaPosMov() ;
		void eaPosRet() ;
		void eaStlPcd() ;
		void eaAnmPcd() ;
		void eaStlRsz() ;
		void eaAnmRsz() ;
		void eaStlZom() ;
		void eaAnmZom() ;
		void eaStlBlr() ;
		void eaAnmBlr() ;
		void eaStlFoc() ;
		void eaAnmFoc() ;
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
		std::stack<State>   h_state_ ;
		std::stack<Event>   h_event_ ;

		// state history values
		std::stack<SmFrame> h_values_ ;

		// event/ action table
		typedef void ( SimpleSM::*EAImp )() ;
		EAImp EATab[static_cast<int>( State::n )][static_cast<int>( Event::n )] = {
			/* S/ E ANM                  BLR                  DIR                  DNS                  FOC                  MOV                  PCD                  POS                  RET                  RSZ                  SCR                  ZOM               */
			/*ANM*/ &SimpleSM::eaReject, &SimpleSM::eaAnmBlr, &SimpleSM::eaAnmDir, &SimpleSM::eaAnmDns, &SimpleSM::eaAnmFoc, &SimpleSM::eaReject, &SimpleSM::eaAnmPcd, &SimpleSM::eaAnmPos, &SimpleSM::eaAnmRet, &SimpleSM::eaAnmRsz, &SimpleSM::eaReject, &SimpleSM::eaAnmZom,
			/*BLR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaBlrRet, &SimpleSM::eaReject, &SimpleSM::eaBlrScr, &SimpleSM::eaReject,
			/*DIR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaDirMov, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaDirRet, &SimpleSM::eaReject, &SimpleSM::eaDirScr, &SimpleSM::eaReject,
			/*FOC*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaFocRet, &SimpleSM::eaReject, &SimpleSM::eaFocScr, &SimpleSM::eaReject,
			/*POS*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaPosMov, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaPosRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*STL*/ &SimpleSM::eaStlAnm, &SimpleSM::eaStlBlr, &SimpleSM::eaStlDir, &SimpleSM::eaStlDns, &SimpleSM::eaStlFoc, &SimpleSM::eaReject, &SimpleSM::eaStlPcd, &SimpleSM::eaStlPos, &SimpleSM::eaStlRet, &SimpleSM::eaStlRsz, &SimpleSM::eaReject, &SimpleSM::eaStlZom,
			/*ZOM*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaZomRet, &SimpleSM::eaReject, &SimpleSM::eaZomScr, &SimpleSM::eaReject
		} ;

		void eaReject() ;

		// group actions
		void eaRdlDns() ;
		void eaRdlDir() ;
		void eaRdlPos() ;
		void eaRdlRsz() ;
} ;

#endif // SIMPLESM_H
