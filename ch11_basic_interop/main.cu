#define VERTEX_SHADER "basic_interop.vert"
#define FRAGMENT_SHADER "basic_interop.frag"
#define _CRT_SECURE_NO_WARNINGS

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <stdlib.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#define NUM_VAOS 3
#define NUM_VBOS 6

// Check for CUDA errors
inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

GLuint vao[NUM_VAOS], vbo[NUM_VBOS];
cudaGraphicsResource* cudaRes[NUM_VBOS];

// Initialize VBO data
__global__ void initVBOs(float4* vbo0, float4* vbo1, float4* vbo2, 
    float4* vbo3, float4* vbo4, float4* vbo5) {

    // Initialize the first VBO
    vbo0[0] = make_float4(-0.15f, -0.15f, 1.00f, -0.15f);
    vbo0[1] = make_float4(0.15f, 1.00f, 0.15f, 0.15f);
    vbo0[2] = make_float4(1.00f, 0.15f, -0.15f, 1.00f);

    // Initialize the second VBO
    vbo1[0] = make_float4(0.00f, 0.00f, 0.00f, 0.25f);
    vbo1[1] = make_float4(0.00f, 0.00f, 0.50f, 0.00f);
    vbo1[2] = make_float4(0.00f, 0.75f, 0.00f, 0.00f);

    // Initialize the third VBO
    vbo2[0] = make_float4(-0.30f, -0.30f, 0.00f, -0.30f);
    vbo2[1] = make_float4(0.30f, 0.00f, 0.30f, 0.30f);
    vbo2[2] = make_float4(0.00f, 0.30f, -0.30f, 0.00f);

    // Initialize the fourth VBO
    vbo3[0] = make_float4(0.00f, 0.00f, 0.00f, 0.00f);
    vbo3[1] = make_float4(0.25f, 0.00f, 0.00f, 0.50f);
    vbo3[2] = make_float4(0.00f, 0.00f, 0.75f, 0.00f);

    // Initialize the fifth VBO
    vbo4[0] = make_float4(-0.45f, -0.45f, -1.00f, -0.45f);
    vbo4[1] = make_float4(0.45f, -1.00f, 0.45f, 0.45f);
    vbo4[2] = make_float4(-1.00f, 0.45f, -0.45f, -1.00f);

    // Initialize the sixth VBO
    vbo5[0] = make_float4(0.00f, 0.00f, 0.00f, 0.00f);
    vbo5[1] = make_float4(0.00f, 0.25f, 0.00f, 0.00f);
    vbo5[2] = make_float4(0.50f, 0.00f, 0.00f, 0.75f);
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

// Initialize buffers
void initBuffers() {

    // Create vertex array objects - one for each square
    glGenVertexArrays(NUM_VAOS, vao);
    glBindVertexArray(vao[0]);

    // Create vertex buffer objects (VBOs) - one for each set of coordinates and colors
    glGenBuffers(NUM_VBOS, vbo);

    // VBO for coordinates of first square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for colors of first square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // VBO for coordinates of second square
    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for colors of second square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    // VBO for coordinates of third square
    glBindVertexArray(vao[2]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for colors of third square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[5]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

// Perform CUDA operations
void runCuda() {

    int i;

    // Register the OpenGL objects
    for (i = 0; i < NUM_VBOS; i++) {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaRes[i], vbo[i],
            cudaGraphicsRegisterFlagsWriteDiscard));
    }

    // Map the OpenGL resources
    cudaGraphicsMapResources(NUM_VBOS, cudaRes, 0);

    // Create memory objects from the VBOs
    float4* floatMem[NUM_VBOS];
    size_t memSize[NUM_VBOS];
    for (i = 0; i < NUM_VBOS; i++) {

        // Initialize the device memory pointer
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&floatMem[i]),
            &memSize[i], cudaRes[i]));
    }

    // Invoke the kernel
    initVBOs<<<1, 1>>> (floatMem[0], floatMem[1],
        floatMem[2], floatMem[3], floatMem[4], floatMem[5]);

    // Unmap the OpenGL resources
    checkCudaErrors(cudaGraphicsUnmapResources(NUM_VBOS, cudaRes, 0));
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(vao[2]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glBindVertexArray(vao[1]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glBindVertexArray(vao[0]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glBindVertexArray(0);
    glutSwapBuffers();
}

// Respond to resize/reshape events
void reshape(int w, int h) {
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
}


// Handle close events
void close() {

    // Unregister CUDA resources
    for (int i = 0; i < NUM_VBOS; i++) {
        checkCudaErrors(cudaGraphicsUnregisterResource(cudaRes[i]));
    }

    // Delete VBOs
    glDeleteBuffers(NUM_VBOS, vbo);

    // Leave the main loop
    glutLeaveMainLoop();
}

int main(int argc, char* argv[]) {

    // Initialize the main window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(300, 300);
    glutCreateWindow("Basic Interoperability");
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

    // Execute CUDA operations
    runCuda();

    // Set callback functions
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutCloseFunc(close);
    glutMainLoop();
    return 0;
}
