#version 330 core

layout (location = 0) in vec3 in_coords;
layout (location = 1) in vec2 in_texcoords;
out vec2 new_texcoords;

void main(void) {

    // Set the texture coordinates
    new_texcoords = in_texcoords;

    // Set the vertex position
    gl_Position = vec4(in_coords, 1.0);
}