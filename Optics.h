#ifndef OPTICS_H
#define OPTICS_H

#include "Ray.h"

struct Payload ;

class Optics {
	public:
		virtual bool spray( const Ray& ray, const Payload& payload, C& attened, Ray& sprayed ) const = 0 ;
} ;

class Diffuse : public Optics {
	public:
		Diffuse( const C& albedo ) : albedo_( albedo ) {}

		virtual bool spray( const Ray& ray, const Payload& payload, C& attened, Ray& sprayed ) const override {
			auto d = payload.normal + rndVon1sphere() ; 
			if ( d.isnear0() )
				d = payload.normal ;

			sprayed = Ray( payload.p, d ) ;
			attened = albedo_ ;

			return true ;
		}

	public:
		C albedo_ ;    // reflectivity
} ;

class Reflect : public Optics {
	public:
		Reflect( const C& albedo, double fuzz ) : albedo_( albedo ), fuzz_( fuzz ) {}

		virtual bool spray( const Ray& ray, const Payload& payload, C& attened, Ray& sprayed ) const override {
			V r = reflect( unitV( ray.dir() ), payload.normal ) ;
			sprayed = Ray( payload.p, r+fuzz_*rndVin1sphere() ) ;
			attened = albedo_ ;

			return ( dot( sprayed.dir(), payload.normal )>0 ) ;
		}

	public:
		C albedo_ ;    // reflectivity
		double fuzz_ ; // fuzz
} ;

class Refract : public Optics {
	public:
		Refract( double refraction_index ) : refraction_index_( refraction_index ) {}

		virtual bool spray( const Ray& ray, const Payload& payload, C& attened, Ray& sprayed ) const override {
			attened = C( 1., 1., 1. ) ;
			V d, u = unitV( ray.dir() ) ;
			double ctta = fmin( dot( -u, payload.normal ), 1. ) ;
			double stta = sqrt( 1.-ctta*ctta ) ;

			double refraction_ratio = payload.facing ? 1./refraction_index_ : refraction_index_ ;
			bool cannot = refraction_ratio*stta>1. ;

			if ( cannot || schlick( ctta, refraction_ratio )>rnd() )
				d = reflect( u, payload.normal ) ;
			else
				d = refract( u, payload.normal, refraction_ratio ) ;

			sprayed = Ray( payload.p, d ) ;
			return true ;
		}

	private:
		double refraction_index_ ;

		// Schlick's reflectance approximation (chapter 10.4)
		static double schlick( double ctta, double rrat ) { auto r0 = ( 1-rrat )/( 1+rrat ) ; r0 = r0*r0 ; return r0+( 1-r0 )*pow( ( 1-ctta ), 5 ) ; }
} ;

#endif
