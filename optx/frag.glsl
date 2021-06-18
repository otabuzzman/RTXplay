#version 330 core

uniform sampler2D t ;

in  vec2 uv ;
out vec4 c ;

void main() {
    c = texture( t, uv ) ;
}
