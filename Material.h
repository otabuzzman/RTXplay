#ifndef MATERIAL_H
#define MATERIAL_H

#include "util.h"

struct record ;

class Material {
	public:
		virtual bool spray( const Ray& ray, const record& rec, C& attened, Ray& sprayed ) const = 0 ;
} ;

class Lambertian : public Material {
	public:
		Lambertian( const C& albedo) : a(albedo) {}

		virtual bool spray( const Ray& ray, const record& rec, C& attened, Ray& sprayed ) const override {
			auto d = rec.normal + rndVon1sphere() ; 
			if ( d.isnear0() )
				d = rec.normal ;

			sprayed = Ray( rec.p, d ) ;
			attened = a ;

			return true ;
		}

	public:
		C a ; // albedo
} ;

class Metal : public Material {
	public:
		Metal( const C& albedo, double fuzz ) : a(albedo), f(fuzz) {}

		virtual bool spray( const Ray& ray, const record& rec, C& attened, Ray& sprayed ) const override {
			V r = reflect( unitV( ray.dir() ), rec.normal ) ;
			sprayed = Ray( rec.p, r+f*rndVin1sphere() ) ;
			attened = a ;

			return ( dot( sprayed.dir(), rec.normal )>0 ) ;
		}

	public:
		C a ;      // albedo
		double f ; // fuzz
} ;

class Dielectric : public Material {
	public:
		Dielectric( double refraction ) : ridx(refraction) {}

		virtual bool spray( const Ray& ray, const record& rec, C& attened, Ray& sprayed ) const override {
			attened = C( 1., 1., 1. ) ;
			V d, u = unitV( ray.dir() ) ;
			double ctta = fmin( dot( -u, rec.normal ), 1. ) ;
			double stta = sqrt( 1.-ctta*ctta ) ;

			double rrat = rec.facing ? 1./ridx : ridx ;
			bool cannot = rrat*stta>1. ;

			if ( cannot || schlick( ctta, rrat )>rnd() )
				d = reflect( u, rec.normal ) ;
			else
				d = refract( u, rec.normal, rrat ) ;

			sprayed = Ray( rec.p, d ) ;
			return true ;
		}

	private:
		double ridx ; // refraction index

		// Schlick's reflectance approximation (chapter 10.4)
		static double schlick( double ctta, double rrat ) { auto r0 = ( 1-rrat )/( 1+rrat ) ; r0 = r0*r0 ; return r0+( 1-r0 )*pow( ( 1-ctta ), 5 ) ; }
} ;

#endif
