#ifndef SIMPLESM_H
#define SIMPLESM_H

#include <stack>

#include "args.h"
#include "camera.h"
#include "thing.h"
#include "rtwo.h"
#include "paddle.h"

// missing in GLFW
extern "C" void glfwGetScroll( GLFWwindow* /*window*/, double* xscroll, double* yscroll ) ;

enum class State { ANM, BLR, DIR, FOC, POS, STL, ZOM, n } ;
static const std::string stateName[] = { "ANM", "BLR", "DIR", "FOC", "POS", "STL", "ZOM" } ;
enum class Event { ANM, BLR, DIR, DNS, FOC, MOV, POS, RET, RSZ, SCR, STL, ZOM, n } ;
static const std::string eventName[] = { "ANM", "BLR", "DIR", "DNS", "FOC", "MOV", "POS", "RET", "RSZ", "SCR", "STL", "ZOM" } ;

struct SmParam {
	LpGeneral             lp_general ;
	GLuint                pbo ;
	cudaGraphicsResource* glx ;

#define SM_OPTION_NONE    0 // default
#define SM_OPTION_ANIMATE 1 // rotate scene around y (WCS)
	unsigned int          options = SM_OPTION_NONE ;
} ;

class SimpleSM {
	public:
		SimpleSM( GLFWwindow* window, const Args& args ) ;
		~SimpleSM() ;

		void transition( const Event& event ) ;

		// event/ action implementations
		void eaStlAnm() ;
		void eaStlRet() ;
		void eaStlDir() ;
		void eaAnmStl() ;
		void eaAnmRet() ;
		void eaAnmDir() ;
		void eaDirScr() ;
		void eaDirMov() ;
		void eaDirRet() ;
		void eaStlPos() ;
		void eaAnmPos() ;
		void eaPosMov() ;
		void eaPosRet() ;
		void eaStlRsz() ;
		void eaStlZom() ;
		void eaStlBlr() ;
		void eaStlFoc() ;
		void eaAnmRsz() ;
		void eaAnmZom() ;
		void eaAnmBlr() ;
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
		std::stack<State> h_state_ ;
		std::stack<Event> h_event_ ;

		// inter-state exchange
		std::stack<int>   i_sexchg_ ;
		// std::stack<float> f_sexchg_ ;
		// std::stack<void*> p_sexchg_ ;

		// event/ action table
		typedef void ( SimpleSM::*EAImp )() ;
		EAImp EATab[static_cast<int>( State::n )][static_cast<int>( Event::n )] = {
			/* S/ E ANM                  BLR                  DIR                  DNS                  FOC                  MOV                  POS                  RET                  RSZ                  SCR                  STL                  ZOM               */
			/*ANM*/ &SimpleSM::eaReject, &SimpleSM::eaAnmBlr, &SimpleSM::eaAnmDir, &SimpleSM::eaReject, &SimpleSM::eaAnmFoc, &SimpleSM::eaReject, &SimpleSM::eaAnmPos, &SimpleSM::eaAnmRet, &SimpleSM::eaAnmRsz, &SimpleSM::eaReject, &SimpleSM::eaAnmStl, &SimpleSM::eaAnmZom,
			/*BLR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaBlrRet, &SimpleSM::eaReject, &SimpleSM::eaBlrScr, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*DIR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaDirMov, &SimpleSM::eaReject, &SimpleSM::eaDirRet, &SimpleSM::eaReject, &SimpleSM::eaDirScr, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*FOC*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaFocRet, &SimpleSM::eaReject, &SimpleSM::eaFocScr, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*POS*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaPosMov, &SimpleSM::eaReject, &SimpleSM::eaPosRet, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*STL*/ &SimpleSM::eaStlAnm, &SimpleSM::eaStlBlr, &SimpleSM::eaStlDir, &SimpleSM::eaReject, &SimpleSM::eaStlFoc, &SimpleSM::eaReject, &SimpleSM::eaStlPos, &SimpleSM::eaStlRet, &SimpleSM::eaStlRsz, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaStlZom,
			/*ZOM*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaZomRet, &SimpleSM::eaReject, &SimpleSM::eaZomScr, &SimpleSM::eaReject, &SimpleSM::eaReject
		} ;

		void eaReject() ;

		// command line arguments
		Args args_ ;
} ;

#endif // SIMPLESM_H
