#define VERTEX_SHADER "three_squares.vert"
#define FRAGMENT_SHADER "three_squares.frag"
#define _CRT_SECURE_NO_WARNINGS

#define NUM_VAOS 3
#define NUM_VBOS 6

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

GLfloat firstCoords[] = { -0.15f, -0.15f, 1.0f,
                          -0.15f,  0.15f, 1.0f,
                           0.15f,  0.15f, 1.0f,
                           0.15f, -0.15f, 1.0f };
GLfloat firstColors[] = { 0.0f,  0.0f, 0.0f,
                          0.25f, 0.0f, 0.0f,
                          0.50f, 0.0f, 0.0f,
                          0.75f, 0.0f, 0.0f };

GLfloat secondCoords[] = { -0.30f, -0.30f, 0.0f,
                           -0.30f,  0.30f, 0.0f,
                            0.30f,  0.30f, 0.0f,
                            0.30f, -0.30f, 0.0f };
GLfloat secondColors[] = { 0.0f, 0.0f,  0.0f,
                           0.0f, 0.25f, 0.0f,
                           0.0f, 0.50f,  0.0f,
                           0.0f, 0.75f, 0.0f };

GLfloat thirdCoords[] = { -0.45f, -0.45f, -1.0f,
                          -0.45f,  0.45f, -1.0f,
                           0.45f,  0.45f, -1.0f,
                           0.45f, -0.45f, -1.0f };
GLfloat thirdColors[] = { 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.25f,
                          0.0f, 0.0f, 0.50f,
                          0.0f, 0.0f, 0.75f };

GLuint vao[NUM_VAOS], vbo[NUM_VBOS];

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

// Compile the given shader object
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

// Initialize the application's buffers
void initBuffers() {

    // Create vertex array objects - one for each square
    glGenVertexArrays(NUM_VAOS, vao);

    // Create vertex buffer objects (VBOs)
    glGenBuffers(NUM_VBOS, vbo);

    // VBO for coordinates of first square
    glBindVertexArray(vao[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), firstCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for colors of first square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), firstColors, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // VBO for coordinates of second square
    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), secondCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for colors of second square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), secondColors, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // VBO for coordinates of third square
    glBindVertexArray(vao[2]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), thirdCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for colors of third square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[5]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), thirdColors, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
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

// Display the content of the GLUT window
void display() {

    // Reset the rendering area
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw the third square
    glBindVertexArray(vao[2]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // Draw the second square
    glBindVertexArray(vao[1]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // Draw the first square
    glBindVertexArray(vao[0]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // Unbind VAOs
    glBindVertexArray(0);

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
    glDeleteBuffers(NUM_VBOS, vbo);

    // Leave the main loop
    glutLeaveMainLoop();
}

int main(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(300, 300);
    glutCreateWindow("Three Squares");
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        perror("Couldn't initialize GLEW");
        exit(1);
    }

    initBuffers();
    initShaders();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutCloseFunc(close);
    glutMainLoop();
    return 0;
}
