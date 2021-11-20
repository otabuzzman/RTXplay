// system includes
#include <iostream>
#include <map>

// subsystem includes
// CUDA
#include <vector_functions.h>

// local includes
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// file specific includes
#include "object.h"

// map comparison function for procWavefrontObj
namespace std {
	bool operator< ( const tinyobj::index_t &l, const tinyobj::index_t &r ) {
		return l.vertex_index<r.vertex_index ;
	}
}

Object::Object( const std::string& wavefront ) {
	procWavefrontObj( wavefront ) ;
}

const Mesh Object::operator[] ( unsigned int i ) {
	return Mesh(
		vces_[i].data(), static_cast<unsigned int>( vces_[i].size() ),
		ices_[i].data(), static_cast<unsigned int>( ices_[i].size() )
	) ;
}
void Object::procWavefrontObj( const std::string& wavefront ) {
	tinyobj::ObjReader       reader ;
	tinyobj::ObjReaderConfig config ;

	std::map<tinyobj::index_t, size_t> recall ; // already seen indices

	reader.ParseFromFile( wavefront, config ) ;
	auto& shapes = reader.GetShapes() ;
	auto& attrib = reader.GetAttrib() ;
	const size_t vertices_size = attrib.vertices.size() ;

	// loop over shapes (submeshes)
	for ( size_t s = 0 ; shapes.size()>s ; s++ ) {
		std::vector<float3> vces ;
		std::vector<uint3>  ices ;

		// loop over faces (triangles)
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
				ices.push_back( { i0, i1, i2 } ) ;

				// fetch triangle vertices
				if ( recall.find( tol_i0 ) == recall.end() ) {
					recall[tol_i0] = vces.size() ;
					vces.push_back( { attrib.vertices[3*i0+0], attrib.vertices[3*i0+1], attrib.vertices[3*i0+2] } ) ;
				}
				if ( recall.find( tol_i1 ) == recall.end() ) {
					recall[tol_i1] = vces.size() ;
					vces.push_back( { attrib.vertices[3*i1+0], attrib.vertices[3*i1+1], attrib.vertices[3*i1+2] } ) ;
				}
				if ( recall.find( tol_i2 ) == recall.end() ) {
					recall[tol_i2] = vces.size() ;
					vces.push_back( { attrib.vertices[3*i2+0], attrib.vertices[3*i2+1], attrib.vertices[3*i2+2] } ) ;
				}
			} else {
				std::ostringstream comment ;

				comment << wavefront << ": triangle faces expected" << std::endl ;
				throw std::runtime_error( comment.str() ) ;
			}
		}

		vces_.push_back( vces ) ;
		ices_.push_back( ices ) ;
	}
}

#ifdef MAIN

int main( const int argc, const char** argv ) {
	Object object( argv[1] ) ;

	std::cout << "o " << argv[1] << std::endl ;

	float3*      vces ;
	unsigned int vces_size ;
	uint3*       ices ;
	unsigned int ices_size ;

	unsigned int v, vsum = 0 ;
	for ( unsigned int o = 0 ; object.size()>o ; o++ ) {
		std::tie( vces, vces_size, ices, ices_size ) = object[o] ;
		for ( v = 0 ; vces_size>v ; v++ )
			printf( "v %f %f %f\n", vces[v].x, vces[v].y, vces[v].z ) ;
		vsum += v ;
	}
	std::cout << "# " << vsum << " vertices"  << std::endl ;

	for ( unsigned int o = 0 ; object.size()>o ; o++ ) {
		std::tie( vces, vces_size, ices, ices_size ) = object[o] ;
		for ( unsigned int i = 0 ; ices_size>i ; i++ )
			printf( "f %d %d %d\n", ices[i].x+1, ices[i].y+1, ices[i].z+1 ) ;
		std::cout << "# " << ices_size << " triangles"  << std::endl ;
	}

	return 0 ;
}

#endif // MAIN
