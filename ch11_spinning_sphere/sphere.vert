#version 330 core

layout (location = 0) in vec4 in_coords;

void main(void) {
	gl_Position = in_coords;
}
