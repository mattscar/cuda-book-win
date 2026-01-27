#version 330 core

in  vec2 new_texcoords;
out vec4 new_color;

uniform sampler2D sam;

void main() {

    // Read the color from the texture
    vec3 color = texture(sam, new_texcoords).rgb;

    // Set the fragment's color
    new_color = vec4(color, 1.0);
}
