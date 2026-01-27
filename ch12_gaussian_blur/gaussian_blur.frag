#version 330 core

in  vec2 new_texcoords;
out vec4 new_color;

uniform sampler2D sam;

void main() {

    // Read the grayscale intensity from the texture
    float gray = texture(sam, new_texcoords).r;

    // Set the fragment's color
    new_color = vec4(gray, gray, gray, 1.0);
}
