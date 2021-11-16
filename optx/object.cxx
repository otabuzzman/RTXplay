// system includes
#include <map>

// subsystem includes
// CUDA
#include <vector_functions.h>

// local includes
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "thing.h"

// file specific includes
#include "object.h"

// initilaize UTM counter
int Object::utm_count_ = 0 ;

// map comparison function for procWavefrontObj
namespace std {
	bool operator< ( const tinyobj::index_t &l, const tinyobj::index_t &r ) {
		return l.vertex_index<r.vertex_index ;
	}
}

Object::Object( const std::string& wavefront ) {
	procWavefrontObj( wavefront ) ;

#ifndef MAIN

	copyVcesToDevice() ;
	num_vces = static_cast<unsigned int>( vces_.size() ) ;
	copyIcesToDevice()  ;
	num_ices = static_cast<unsigned int>( ices_.size() ) ;

#endif // MAIN

	// count UTM and record current
	utm_index = utm_count_++ ;
}

Object::~Object() noexcept ( false ) {
#ifndef MAIN

	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( Thing::vces ) ) ) ;
	CUDA_CHECK( cudaFree( reinterpret_cast<void*>( Thing::ices ) ) ) ;

#endif // MAIN
}

void Object::procWavefrontObj( const std::string& wavefront ) {
	tinyobj::ObjReader       reader ;
	tinyobj::ObjReaderConfig config ;

	std::map<tinyobj::index_t, size_t> recall ; // already seen indices

	reader.ParseFromFile( wavefront, config ) ;
	auto& shapes = reader.GetShapes() ;
	auto& attrib = reader.GetAttrib() ;
	const size_t vertices_size = attrib.vertices.size() ;

	// loop over shapes
	for ( size_t s = 0 ; shapes.size()>s ; s++ ) {
		// loop over faces
		for ( size_t f = 0 ; shapes[s].mesh.num_face_vertices.size()>f ; f++ ) {
			// triangle check
			if ( shapes[s].mesh.num_face_vertices[f] == 3 ) {
				// fetch triangle indices
				tinyobj::index_t tol_i0 = shapes[s].mesh.indices[3*f+0] ;
				tinyobj::index_t tol_i1 = shapes[s].mesh.indices[3*f+1] ;
				tinyobj::index_t tol_i2 = shapes[s].mesh.indices[3*f+2] ;
				unsigned int i0 = tol_i0.vertex_index ;
				unsigned int i1 = tol_i1.vertex_index ;
				unsigned int i2 = tol_i2.vertex_index ;
				if ( i0>vertices_size || i1>vertices_size || i2>vertices_size ) {
					std::ostringstream comment ;

					comment << wavefront << ": index out of bounds" << std::endl ;
					throw std::runtime_error( comment.str() ) ;
				}
				ices_.push_back( { i0, i1, i2 } ) ;

				// fetch triangle vertices
				if ( recall.find( tol_i0 ) == recall.end() ) {
					recall[tol_i0] = vces_.size() ;
					vces_.push_back( { attrib.vertices[3*i0+0], attrib.vertices[3*i0+1], attrib.vertices[3*i0+2] } ) ;
				}
				if ( recall.find( tol_i1 ) == recall.end() ) {
					recall[tol_i1] = vces_.size() ;
					vces_.push_back( { attrib.vertices[3*i1+0], attrib.vertices[3*i1+1], attrib.vertices[3*i1+2] } ) ;
				}
				if ( recall.find( tol_i2 ) == recall.end() ) {
					recall[tol_i2] = vces_.size() ;
					vces_.push_back( { attrib.vertices[3*i2+0], attrib.vertices[3*i2+1], attrib.vertices[3*i2+2] } ) ;
				}
			} else {
				std::ostringstream comment ;

				comment << wavefront << ": triangle faces expected" << std::endl ;
				throw std::runtime_error( comment.str() ) ;
			}
		}
	}
}

void Object::copyVcesToDevice() {
	const size_t vces_size = sizeof( float3 )*static_cast<unsigned int>( vces_.size() ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &( Thing::vces ) ), vces_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
		reinterpret_cast<void*>( Thing::vces ),
		vces_.data(),
		vces_size,
		cudaMemcpyHostToDevice
		) ) ;
}

void Object::copyIcesToDevice() {
	const size_t ices_size = sizeof( uint3 )*static_cast<unsigned int>( ices_.size() ) ;
	CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &( Thing::ices ) ), ices_size ) ) ;
	CUDA_CHECK( cudaMemcpy(
		reinterpret_cast<void*>( Thing::ices ),
		ices_.data(),
		ices_size,
		cudaMemcpyHostToDevice
		) ) ;
}

#ifdef MAIN

int main( const int argc, const char** argv ) {
	Object object( argv[1] ) ;

	std::cout << "o " << argv[1] << std::endl ;
	for ( auto& v : object.vces() )
		printf( "v %f %f %f\n", v.x, v.y, v.z ) ;
	for ( auto& i : object.ices() )
		printf( "f %u %u %u\n", i.x+1, i.y+1, i.z+1 ) ;

	return 0 ;
}

#endif // MAIN
