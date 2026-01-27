#version 330 core

layout (location = 0) in vec3 in_coords;
layout (location = 1) in vec3 in_color;
out vec3 new_color;

void main(void) {

    // Set the vertex color
	new_color = in_color;

    // Create a rotation matrix
    mat3x3 rot_matrix = mat3x3(0.707, 0.641, -0.299,
                             -0.707, 0.641, -0.299,
                             -0.000, 0.423,  0.906);

    // Rotate the vertex
    vec3 coords = rot_matrix * in_coords;

    // Set the vertex position
	gl_Position = vec4(coords, 1.0);
}
