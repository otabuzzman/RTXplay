#ifndef SIMPLESM_H
#define SIMPLESM_H

// system includes
#include <stack>
#include <string>

// subsystem includes
// GLAD
#include <glad/glad.h>
// GLFW
#include <GLFW/glfw3.h>
// CUDA
#include <cuda_gl_interop.h> // must follow glad.h

// local includes
#include "args.h"
#include "denoiser.h"
#include "paddle.h"

// file specific includes
// none

// missing in GLFW
extern "C" void glfwGetScroll( GLFWwindow* /*window*/, double* xscroll, double* yscroll ) ;

enum class State { ANM, BLR, DIR, EDT, FOC, ODI, OPO, POS, SCL, STL, ZOM, n } ;
static const std::string state_name[] = { "ANM", "BLR", "DIR", "EDT", "FOC", "ODI", "OPO", "POS", "SCL", "STL", "ZOM" } ;
enum class Event { ANM, BLR, DIR, DNS, EDT, FOC, MOV, PCD, POS, RET, RSZ, SCL, SCR, ZOM, n } ;
static const std::string event_name[] = { "ANM", "BLR", "DIR", "DNS", "EDT", "FOC", "MOV", "PCD", "POS", "RET", "RSZ", "SCL", "SCR", "ZOM" } ;

struct SmParam {
	GLuint                pbo ;
	cudaGraphicsResource* glx ;

	void                  ( *glfwPoWaEvents )() = &glfwWaitEvents ;

	OptixDeviceContext*   optx_context = nullptr ;
	Denoiser*             denoiser     = nullptr ;

	unsigned int          pick_id ;
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
		void eaStlEdt() ;
		void eaAnmDns() ;
		void eaAnmEdt() ;
		void eaStlRet() ;
		void eaAnmRet() ;
		void eaStlDir() ;
		void eaAnmDir() ;
		void eaDirScr() ;
		void eaDirMov() ;
		void eaDirRet() ;
		void eaStlPos() ;
		void eaAnmPos() ;
		void eaPosScr() ;
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
		void eaEdtPos() ;
		void eaEdtDir() ;
		void eaEdtRet() ;
		void eaOpoRet() ;
		void eaOdiRet() ;
		void eaOpoMov() ;
		void eaOdiMov() ;
		void eaOpoScl() ;
		void eaOpoScr() ;
		void eaOdiScr() ;
		void eaSclScr() ;
		void eaSclRet() ;

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
			/* S/ E ANM                  BLR                  DIR                  DNS                  EDT                  FOC                  MOV                  PCD                  POS                  RET                  RSZ                  SCL                  SCR                  ZOM               */
			/*ANM*/ &SimpleSM::eaReject, &SimpleSM::eaAnmBlr, &SimpleSM::eaAnmDir, &SimpleSM::eaAnmDns, &SimpleSM::eaAnmEdt, &SimpleSM::eaAnmFoc, &SimpleSM::eaReject, &SimpleSM::eaAnmPcd, &SimpleSM::eaAnmPos, &SimpleSM::eaAnmRet, &SimpleSM::eaAnmRsz, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaAnmZom,
			/*BLR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaBlrRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaBlrScr, &SimpleSM::eaReject,
			/*DIR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaDirMov, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaDirRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaDirScr, &SimpleSM::eaReject,
			/*EDT*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaEdtDir, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaEdtPos, &SimpleSM::eaEdtRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*FOC*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaFocRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaFocScr, &SimpleSM::eaReject,
			/*ODI*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaOdiMov, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaOdiRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaOdiScr, &SimpleSM::eaReject,
			/*OPO*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaOpoMov, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaOpoRet, &SimpleSM::eaReject, &SimpleSM::eaOpoScl, &SimpleSM::eaOpoScr, &SimpleSM::eaReject,
			/*POS*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaPosMov, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaPosRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaPosScr, &SimpleSM::eaReject,
			/*SCL*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaSclRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaSclScr, &SimpleSM::eaReject,
			/*STL*/ &SimpleSM::eaStlAnm, &SimpleSM::eaStlBlr, &SimpleSM::eaStlDir, &SimpleSM::eaStlDns, &SimpleSM::eaStlEdt, &SimpleSM::eaStlFoc, &SimpleSM::eaReject, &SimpleSM::eaStlPcd, &SimpleSM::eaStlPos, &SimpleSM::eaStlRet, &SimpleSM::eaStlRsz, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaStlZom,
			/*ZOM*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaZomRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaZomScr, &SimpleSM::eaReject
		} ;

		void eaReject() ;

		// shared group actions called in different states on same event
		void eaRdlDns() ; // RDL state group
		void eaRdlDir() ;
		void eaRdlPos() ;
		void eaRdlRsz() ;
		// shared group actions called in single state on different events
		void eaEdtSed() ; // SED state group
} ;

#endif // SIMPLESM_H
