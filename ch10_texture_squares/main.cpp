#define VERTEX_SHADER "texture_squares.vert"
#define FRAGMENT_SHADER "texture_squares.frag"
#define _CRT_SECURE_NO_WARNINGS

#define NUM_SQUARES 3
#define TEXTURE_1 "checker.png"
#define TEXTURE_2 "square.png"
#define TEXTURE_3 "stripe.png"

#pragma warning( disable : 4189 )
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

GLfloat firstCoords[] = { -0.15f, -0.15f, 1.0f,
                          -0.15f,  0.15f, 1.0f,
                           0.15f,  0.15f, 1.0f,
                           0.15f, -0.15f, 1.0f };

GLfloat secondCoords[] = { -0.30f, -0.30f, 0.0f,
                           -0.30f,  0.30f, 0.0f,
                            0.30f,  0.30f, 0.0f,
                            0.30f, -0.30f, 0.0f };

GLfloat thirdCoords[] = { -0.45f, -0.45f, -1.0f,
                          -0.45f,  0.45f, -1.0f,
                           0.45f,  0.45f, -1.0f,
                           0.45f, -0.45f, -1.0f };

GLfloat texCoords[] = { 0.0f, 0.0f,
                        0.0f, 1.0f,
                        1.0f, 1.0f,
                        1.0f, 0.0f };

GLuint vao[NUM_SQUARES], vbo[NUM_SQUARES * 2], textures[NUM_SQUARES];

char* readFile(const char* filename, GLint* size) {

    FILE* handle;
    char* buffer;

    // Read program file and place content into buffer
    handle = fopen(filename, "r");
    if (handle == NULL) {
        perror("Couldn't find the file");
        exit(1);
    }
    fseek(handle, 0, SEEK_END);
    *size = ftell(handle);
    rewind(handle);
    buffer = (char*)malloc(*size + 1);
    buffer[*size] = '\0';
    fread(buffer, sizeof(char), *size, handle);
    fclose(handle);

    return buffer;
}

void compileShader(GLint shader) {

    GLint success;
    GLsizei log_size;
    GLchar* log;

    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_size);
        log = (char*)malloc(log_size + 1);
        log[log_size] = '\0';
        glGetShaderInfoLog(shader, log_size + 1, NULL, log);
        printf("%s\n", log);
        free(log);
        exit(1);
    }
}

void initBuffers() {

    // Create a vertex array for each square
    glGenVertexArrays(NUM_SQUARES, vao);

    // Create 6 vertex buffer objects (VBOs) - one for each set of coordinates and colors
    glGenBuffers(NUM_SQUARES * 2, vbo);

    // VBO for coordinates of first square
    glBindVertexArray(vao[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), firstCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for texture coordinates of first square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // VBO for coordinates of second square
    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), secondCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for texture coordinates of second square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // VBO for coordinates of third square
    glBindVertexArray(vao[2]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), thirdCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for texture coordinates of third square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[5]);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// Initialize the shaders
void initShaders() {

    GLuint vs, fs, prog;
    char *vsSource, *fsSource;
    GLint vsLength, fsLength;

    // Create shader objects
    vs = glCreateShader(GL_VERTEX_SHADER);
    fs = glCreateShader(GL_FRAGMENT_SHADER);

    // Read shader code from the source files
    vsSource = readFile(VERTEX_SHADER, &vsLength);
    fsSource = readFile(FRAGMENT_SHADER, &fsLength);

    // Associate shader code with the shader objects
    glShaderSource(vs, 1, (const char**)&vsSource, &vsLength);
    glShaderSource(fs, 1, (const char**)&fsSource, &fsLength);

    // Compile the shaders
    compileShader(vs);
    compileShader(fs);

    // Create the program
    prog = glCreateProgram();

    // Attach the shaders to the program
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);

    // Link the program
    glLinkProgram(prog);
    glUseProgram(prog);
}

void initTextures() {

    int width[NUM_SQUARES], height[NUM_SQUARES], num_channels;
    const char* tex_names[NUM_SQUARES] = { TEXTURE_1, TEXTURE_2, TEXTURE_3 };
    int i;

    glEnable(GL_TEXTURE_2D);
    glGenTextures(NUM_SQUARES, textures);

    for (i = 0; i < NUM_SQUARES; i++) {

        // Make texture active
        glBindTexture(GL_TEXTURE_2D, textures[i]);

        // Set texture parameters
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        // Read pixel data and associate it with texture
        unsigned char* tex_pixels = stbi_load(tex_names[i], &width[i], &height[i], &num_channels, 3);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width[i], height[i],
            0, GL_RGB, GL_UNSIGNED_BYTE, tex_pixels);

        // Free pixel data
        stbi_image_free(tex_pixels);
    }
}

void display() {

    // Reset the rendering area
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw the third square
    glBindVertexArray(vao[2]);
    glBindTexture(GL_TEXTURE_2D, textures[2]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // Draw the second square
    glBindVertexArray(vao[1]);
    glBindTexture(GL_TEXTURE_2D, textures[1]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // Draw the first square
    glBindVertexArray(vao[0]);
    glBindTexture(GL_TEXTURE_2D, textures[0]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // Unbind VAOs
    glBindVertexArray(0);
    glFinish();

    // Swap buffers
    glutSwapBuffers();
}

// Handle reshape/resize events
void reshape(int w, int h) {
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
}

// Handle close events
void close() {

    // Delete VBOs
    glDeleteBuffers(NUM_SQUARES * 2, vbo);

    // Delete textures
    glDeleteTextures(NUM_SQUARES, textures);

    // Leave the main loop
    glutLeaveMainLoop();
}

int main(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(300, 300);
    glutCreateWindow("Texture Squares");
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        perror("Couldn't initialize GLEW");
        exit(1);
    }

    initBuffers();
    initShaders();
    initTextures();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutCloseFunc(close);
    glutMainLoop();
    return 0;
}
