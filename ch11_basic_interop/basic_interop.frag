#version 330 core

in  vec3 new_color;
out vec4 out_color;

void main(void) {

    // Update the color
	vec3 tmp_color = new_color + vec3(0.25f, 0.25f, 0.25f);

	// Set the fragment's color
	out_color = vec4(tmp_color, 1.0);
}
