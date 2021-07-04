#ifndef SIMPLESM_H
#define SIMPLESM_H

#include <stack>

#include "camera.h"
#include "paddle.h"
#include "thing.h"
#include "rtwo.h"

const enum class State { BLR, CTL, DIR, FOC, POS, ZOM, n } ;
const std::string stateName[] = { "BLR", "CTL", "DIR", "FOC", "POS", "ZOM" } ;
const enum class Event { BLR, DIR, FOC, MOV, POS, RET, RSZ, SCR, ZOM, n } ;
const std::string eventName[] = { "BLR", "DIR", "FOC", "MOV", "POS", "RET", "RSZ", "SCR", "ZOM" } ;

class SimpleSM {
	public:
		SimpleSM( GLFWwindow* window, LpGeneral* lp_general ) ;

		void transition( const Event event ) ;

		// event/ action implementations
		void eaCtlRet() ;
		void eaCtlDir() ;
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
		LpGeneral* lp_general_ ;
		std::stack<State> h_state_ ; // state history
		std::stack<Event> h_event_ ; // event history
		Paddle paddle ;

		// event/ action table
		typedef void ( SimpleSM::*EAImp )() ;
		EAImp EATab[static_cast<int>( State::n )][static_cast<int>( Event::n )] = {
			/* S/ E BLR                  DIR                  FOC                  MOV                  POS                  RET                  RSZ                  SCR                  ZOM               */
			/*BLR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*CTL*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*DIR*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*FOC*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*POS*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject,
			/*ZOM*/ &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject, &SimpleSM::eaReject
		} ;

		void eaReject() ;
} ;

#endif // SIMPLESM_H
