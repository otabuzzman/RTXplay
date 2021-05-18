#version 330 core

layout ( location = 0 ) in vec3 v ;

out vec2 uv ;

void main() {
	gl_Position = vec4( v.x, v.y, v.z, 1. ) ;
	uv = ( vec2( v.x, v.y )+vec2( 1., 1. ) )/2. ;
}
