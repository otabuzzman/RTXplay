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

const Mesh Object::operator[] ( unsigned int m ) {
	return Mesh(
		vces_.data(),    static_cast<unsigned int>( vces_.size() ),
		ices_[m].data(), static_cast<unsigned int>( ices_[m].size() )
	) ;
}

void Object::procWavefrontObj( const std::string& wavefront ) {
	tinyobj::ObjReader       reader ;
	tinyobj::ObjReaderConfig config ;
	config.mtl_search_path = "./" ;

	std::map<tinyobj::index_t, size_t> recall ; // already seen indices

	if ( ! reader.ParseFromFile( wavefront, config ) ) {
		std::ostringstream comment ;

		comment << wavefront << ": could not parse" << std::endl ;
		throw std::runtime_error( comment.str() ) ;
	}

	auto& shapes = reader.GetShapes() ;
	auto& attrib = reader.GetAttrib() ;
	const size_t vertices_size = attrib.vertices.size() ;
	auto& materials = reader.GetMaterials();

	// loop over shapes (submeshes)
	for ( size_t s = 0 ; shapes.size()>s ; s++ ) {
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

			unsigned int m = shapes[s].mesh.material_ids[f] ;
			if ( m == -1 )
				continue ; // shape without material

			// RTOW: albedo
			const float3 Kd = {
				materials[m].diffuse[0],
				materials[m].diffuse[1],
				materials[m].diffuse[2]
			} ;
			// RTOW: index
			const float Ni  = materials[m].ior ;
			// RTOW: fuzz
			std::map<std::string, std::string>::const_iterator kv = materials[m].unknown_parameter.find( "sharpness" ) ;
			unsigned int sharpness = kv == materials[m].unknown_parameter.end() ? 60 : sscanf( kv->second.c_str(), "%u", &sharpness ) ;
		}

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

	std::tie( vces, vces_size, ices, ices_size ) = object[0] ;
	for ( unsigned int v = 0 ; vces_size>v ; v++ )
		printf( "v %f %f %f\n", vces[v].x, vces[v].y, vces[v].z ) ;
	std::cout << "# " << vces_size << " vertices" << std::endl ;

	for ( unsigned int o = 0 ; object.size()>o ; o++ ) {
		std::tie( vces, vces_size, ices, ices_size ) = object[o] ;
		for ( unsigned int i = 0 ; ices_size>i ; i++ )
			printf( "f %d %d %d\n", ices[i].x+1, ices[i].y+1, ices[i].z+1 ) ;
		std::cout << "# " << ices_size << " triangles" << std::endl ;
	}

	return 0 ;
}

#endif // MAIN
