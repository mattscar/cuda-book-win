#define VERTEX_SHADER "sphere.vert"
#define FRAGMENT_SHADER "sphere.frag"
#define _CRT_SECURE_NO_WARNINGS

#define NUM_VERTICES 256

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <stdlib.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Check for CUDA errors
inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

GLuint vao, vbo;
cudaGraphicsResource* cudaRes;
float tick = 0.0f;

// Initialize VBO data
__global__ void sphere(float4* vertices, float tick) {

    // Compute latitude and longitude
    int id = threadIdx.x;
    int longitude = id / 16;
    int latitude = id % 16;

    // Compute spherical coordinates
    float radius = 0.75;
    float sign = -2.0f * (longitude % 2) + 1.0f;
    float phi = 2.0f * CUDART_PI_F * longitude / 16 + tick;
    float theta = CUDART_PI_F * latitude / 16;

    // Compute rectangular coordinates
    vertices[id].x = radius * sin(theta) * cos(phi);
    vertices[id].y = radius * sign * cos(theta);
    vertices[id].z = radius * sin(theta) * sin(phi);
    vertices[id].w = 1.0f;
}

// Read a character buffer from a file
char* readFile(const char* filename, GLint* size) {

    FILE* handle;
    char* buffer;

    // Read program file and place content into buffer
    handle = fopen(filename, "r");
    if (handle == NULL) {
        std::cerr << "Couldn't find the file" << std::endl;
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

// Compile the shader
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
        std::cout << log << std::endl;
        free(log);
        exit(1);
    }
}

// Initialize the shaders
void initShaders() {

    GLuint vs, fs, prog;
    char* vsSource, * fsSource;
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

// Initialize buffers
void initBuffers() {

    // Create vertex array objects
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create vertex buffers
    glGenBuffers(1, &vbo);

    // VBO containing sphere coordinates
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 4 * NUM_VERTICES * sizeof(GLfloat),
        NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}

// Perform CUDA operations
void runCuda() {

    // Register the OpenGL object
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaRes, vbo,
        cudaGraphicsRegisterFlagsWriteDiscard));

    // Map the OpenGL resource
    cudaGraphicsMapResources(1, &cudaRes, 0);

    // Create memory objects from the VBOs
    float4* floatMem;
    size_t memSize;

    // Initialize the device memory pointer
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&floatMem),
        &memSize, cudaRes));

    // Invoke the kernel
    sphere<<<1, NUM_VERTICES>>>(floatMem, tick);

    // Unmap the OpenGL resources
    cudaGraphicsUnmapResources(1, &cudaRes, 0);
}

void display() {

    // Clear the rendering surface
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Launch kernel to update VBO
    runCuda();

    // Render model with updated VBO
    glBindVertexArray(vao);

    // Draw vertices in a line loop
    glDrawArrays(GL_LINE_LOOP, 0, NUM_VERTICES);

    // Update timing
    tick += 0.001f;

    // Unbind the vertex array object
    glBindVertexArray(0);

    // Make the framebuffer image visible
    glutSwapBuffers();

    // Restart display
    glutPostRedisplay();
}

// Respond to resize/reshape events
void reshape(int w, int h) {
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
}


// Handle close events
void close() {

    // Unregister CUDA resource
    checkCudaErrors(cudaGraphicsUnregisterResource(cudaRes));

    // Delete VBOs
    glDeleteBuffers(1, &vbo);

    // Leave the main loop
    glutLeaveMainLoop();
}

int main(int argc, char* argv[]) {

    // Initialize the main window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(300, 300);
    glutCreateWindow("Spinning Sphere");
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // Launch GLEW processing
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Couldn't initialize GLEW" << std::endl;
        exit(1);
    }

    // Create GL data objects
    initBuffers();

    // Create and compile shaders
    initShaders();

    // Set callback functions
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutCloseFunc(close);
    glutMainLoop();
    return 0;
}
