#define VERTEX_SHADER "gaussian_blur.vert"
#define FRAGMENT_SHADER "gaussian_blur.frag"
#define _CRT_SECURE_NO_WARNINGS

#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <stdlib.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

// Check for CUDA errors
inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

int imgWidth, imgHeight;
GLuint vao, vbo[2], texture;
cudaArray_t inputImage, outputImage;
cudaGraphicsResource* cudaRes;
GLfloat vertexCoords[] = { -1.0f, -1.0f, 0.0f,
                           -1.0f,  1.0f, 0.0f,
                            1.0f,  1.0f, 0.0f,
                            1.0f, -1.0f, 0.0f };

GLfloat texCoords[] = { 0.0f, 1.0f,
                        0.0f, 0.0f,
                        1.0f, 0.0f,
                        1.0f, 1.0f };

// Set coefficients of the Gaussian blur
__constant__ float GAUSS_KERNEL[3][3] = {
    {1.f / 16.f, 2.f / 16.f, 1.f / 16.f},
    {2.f / 16.f, 4.f / 16.f, 2.f / 16.f},
    {1.f / 16.f, 2.f / 16.f, 1.f / 16.f}
};

// Perform the Gaussian blur
__global__ void gaussianBlur(cudaTextureObject_t texObj, 
    cudaSurfaceObject_t surfObj, int width, int height) {

    // Get pixel location for the thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Use normalized coordinates for texture sampling
    float invW = 1.0f / width;
    float invH = 1.0f / height;
    float sum = 0.0f;

    // 3x3 Gaussian convolution
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            float u = (x + 0.5f + kx) * invW;
            float v = (y + 0.5f + ky) * invH;
            float pixel = tex2D<float>(texObj, u, v); // grayscale sample
            sum += pixel * GAUSS_KERNEL[ky + 1][kx + 1];
        }
    }

    // Convert to 8-bit and write result
    unsigned char outVal = static_cast<unsigned char>(fminf(sum * 255.0f, 255.0f));
    surf2Dwrite(outVal, surfObj, x * sizeof(unsigned char), y);
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

    // Create a vertex array
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create two vertex buffer objects (VBOs)
    // One for vertex coordinates, one for texture coordinates
    glGenBuffers(2, vbo);

    // VBO for coordinates of first square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(GLfloat), vertexCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // VBO for texture coordinates of first square
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), texCoords, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// Configure and intialize texture
void initTextures() {

    // Create OpenGL texture
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);

    // Make texture active
    glBindTexture(GL_TEXTURE_2D, texture);

    // Read pixel data and associate it with texture
    int numChannels;
    unsigned char* imgData = stbi_load("noisy.png", &imgWidth, 
        &imgHeight, &numChannels, 1);

    // Allocate memory for the texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, imgWidth, imgHeight, 0,
        GL_RED, GL_UNSIGNED_BYTE, NULL);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Describe the image channel
    cudaChannelFormatDesc desc =
        cudaCreateChannelDesc<unsigned char>();

    // Allocate memory for the CUDA array
    checkCudaErrors(cudaMallocArray(&inputImage, &desc, imgWidth, imgHeight));

    // Copy image data to CUDA array
    checkCudaErrors(cudaMemcpy2DToArray(inputImage, 0, 0, imgData, imgWidth * sizeof(unsigned char),
        imgWidth * sizeof(unsigned char), imgHeight, cudaMemcpyHostToDevice));

    // Free pixel data
    stbi_image_free(imgData);
}

// Perform CUDA operations
void runCuda() {

    // Create the resource descriptor
    cudaResourceDesc texRes = {};
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = inputImage;

    // Create the texture descriptor
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    // Create the texture object
    cudaTextureObject_t texObj;
    checkCudaErrors(cudaCreateTextureObject(&texObj, &texRes, &texDesc, nullptr));

    // Register the OpenGL texture
    checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaRes, texture, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore));

    // Map the CUDA resource
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaRes, 0));

    // Access the CUDA array for the CUDA resource
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&outputImage, cudaRes, 0, 0));

    // Create the resource descriptor for the surface object
    cudaResourceDesc surfRes = {};
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = outputImage;

    // Create the surface object
    cudaSurfaceObject_t surfObj;
    checkCudaErrors(cudaCreateSurfaceObject(&surfObj, &surfRes));

    // Invoke the kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((imgWidth + blockSize.x - 1) / blockSize.x,
        (imgHeight + blockSize.y - 1) / blockSize.y);
    gaussianBlur<<<gridSize, blockSize>>>(texObj, surfObj, imgWidth, imgHeight);

    // Wait for the kernel to finish
    checkCudaErrors(cudaDeviceSynchronize());

    // Destroy the texture and surface object
    checkCudaErrors(cudaDestroySurfaceObject(surfObj));
    checkCudaErrors(cudaDestroyTextureObject(texObj));

    // Unmap the CUDA resource
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaRes, 0));
}

void display() {

    // Clear the rendering surface
    glClear(GL_COLOR_BUFFER_BIT);

    // Render model with updated VBO
    glBindVertexArray(vao);

    // Draw vertices in a line loop
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    // Unbind the vertex array object
    glBindVertexArray(0);

    // Make the framebuffer image visible
    glutSwapBuffers();
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
    glDeleteBuffers(2, vbo);

    // Delete texture
    glDeleteTextures(1, &texture);

    // Leave the main loop
    glutLeaveMainLoop();
}

int main(int argc, char* argv[]) {

    // Initialize the main window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(300, 300);
    glutCreateWindow("Gaussian Blur");
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

    // Configure and initialize textures
    initTextures();

    // Run CUDA operations
    runCuda();

    // Set callback functions
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutCloseFunc(close);
    glutMainLoop();
    return 0;
}
