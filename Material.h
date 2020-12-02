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
};

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
};

#endif
