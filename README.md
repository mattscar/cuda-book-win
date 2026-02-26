# Example code for *CUDA 13: High-Performance Computing and Graphics*

This repository contains the example projects for the book *CUDA 13: High-Performance Computing and Graphics* by Matt Scarpino.

The projects in the chapters are given as follows:

- **ch02_intro** - Simple application to verify installation of the CUDA toolkit
- **ch03_add_arrays** - Adds arrays of values together, demonstrates memory operations
- **ch03_check_properties** - Reads processing capabilities of the graphics card
- **ch04_half_floats** - Demonstrates usage of half-precision floating-point values
- **ch04_memory_spaces** - Shows how different memory spaces can be accessed in CUDA
- **ch04_print_builtin** - Displays values of CUDA's built-in variables
- **ch04_shared_sync** - Demonstrates usage of shared memory and thread block synchronization
- **ch05_mapped_memory** - Accelerates data access by taking advantage of mapped memory
- **ch05_stream_events** - Shows how CUDA streams and events work together
- **ch05_tree_recursion** - Uses dynamic parallelism to implement recursion
- **ch05_wmma_demo** - Performs matrix multiplication and accumulation using the WMMA API
- **ch05_complex_dot** - Performs a complex-valued dot product using cuBLAS
- **ch06_matrix_vec** - Uses cuBLAS to multiply a matrix by a vector
- **ch07_dense_solver** - Demonstrates how cuSOLVER can be used to solve a dense matrix system
- **ch07_sparse_solver** - Shows how cuDSS can be used to solve a sparse matrix system
- **ch08_fft_ifft** - Performs the forward FFT and inverse FFT using the cuFFT package
- **ch09_image_constrast** - Updates an image's contrast using the NPPI package
- **ch09_image_filter** - Performs image filtering using NPPI
- **ch09_rotate_resize** - Rotates and resizes an image using NPPI
- **ch10_texture_squares** - Applies texture to an OpenGL rendering
- **ch10_three_squares** - Creates a simple OpenGL rendering
- **ch11_basic_interop** - Demonstrates how CUDA and OpenGL can work together
- **ch11_spinning_sphere** - Animates rendering using CUDA and OpenGL
- **ch12_gaussian_blur** - Shows how CUDA and OpenGL can work together to filter textures


The example projects in the appendices are:

- **appa_add_arrays** - Uses the CUDA Driver API to add two arrays
- **appa_device_attributes** - Accesses device attributes using the CUDA Driver API
- **appa_runtime_compile** - Demonstrates how NVRTC makes runtime compilation possible
- **appb_array_reverse** - Reverses an array of values using Parallel Thread Execution (PTX)
- **appb_simple_ptx** - Presents a simple application based on Parallel Thread Execution (PTX)

This example code is provided in a Visual Studio solution containing twenty-nine projects. Each project has a file named <proj_name>.vcxproj that defines the project's properties. Each file reads the `CUDA_HOME` variable and creates variables inside a property group:

```
<PropertyGroup>
  <CudaPathClean Condition="'$(CUDA_PATH)' != ''">
    $(CUDA_PATH.TrimEnd('\'))
  </CudaPathClean>
  <CudaVersionDir Condition="'$(CudaPathClean)' != ''">
    $([System.IO.Path]::GetFileName($(CudaPathClean)))
  </CudaVersionDir>
  <DetectedCudaVer Condition="'$(CudaVersionDir)' != ''">
    $(CudaVersionDir.Replace('v', ''))
  </DetectedCudaVer>
</PropertyGroup>
```

After defining these properties, the project file can access the properties of the CUDA Toolkit with the following import group definition:

```
<ImportGroup Label="ExtensionSettings">
  <Import Project="$(VCTargetsPath)\BuildCustomizations\
    CUDA $(DetectedCudaVer).props" Condition="'$(DetectedCudaVer)' 
    != '' And Exists('$(VCTargetsPath)\BuildCustomizations\CUDA 
    $(DetectedCudaVer).props')" />
</ImportGroup>
```

This accesses properties for the latest installed CUDA Toolkit. As a result, you can specify CUDA library names and header files without having to enter the toolkit's path. 

For this book, the most important CUDA library is cudart.lib, which provides the CUDA Runtime. This is a shared library, and when an executable runs, it needs to access cudart64_13.dll in the toolkit's bin\x64 folder. If you get any errors related to this, it's important to make sure the DLL's location is part of your PATH variable.

If you right-click the solution and select Build Solution, Visual Studio will build most of the projects. By default, it doesn't build the projects that require cuDSS and OpenGL. cuDSS installation is explained in Chapter 7 and OpenGL installation is explained in Chapter 10. 

Once these packages are installed, you can add these projects to the solution build by right-clicking the solution and selecting Properties. In the dialog that appears, go to the Configuration item and check the unchecked projects in the Build column.

If a project's build succeeds, the executable and required files will be stored in the solution's x64\Debug folder. If you open a command prompt and change to this directory, you'll be able to run each executable from the command line.
