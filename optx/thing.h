#ifndef THING_H
#define THING_H

class Thing {
	public:
		const std::vector<float3> vces() const ;
		const std::vector<uint3>  ices() const ;

	protected:
		std::vector<float3> vces_ ;
		std::vector<uint3>  ices_ ;
} ;

#endif // THING_H
