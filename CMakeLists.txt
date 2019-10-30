## CMAKE_DOCUMENTATION_START CMakeLists.txt
##
## Main CMakeFile for compiling RayTraceDicom.
## Following variables can be configured when running ccmake:
## <table>
## <caption id="config-cmake">Table of configurable CMake parameters</caption>
## <tr><th>Variable             <th>Values                  <th>Description
## <tr><td>CMAKE_BUILD_TYPE     <td>Release (Debug)         <td>Choose the type of build
## <tr><td>WATER_CUBE_TEST      <td>OFF (ON)                <td>Use a water cube as test phantom, ignore IMG_DATA_DIRECTORY
## <tr><td>DOSE_TO_WATER        <td>ON (OFF)                <td>Calculate dose to water
## <tr><td>FINE_GRAINED_TIMING  <td>OFF (ON)                <td>Unset it to get more accurate total time, reports up to 30 ms longer execution time
## <tr><td>NOZZLE               <td>ON (OFF)                <td>Take into account air in the nozzle for sigma sq calculation
## <tr><td>NUCLEAR_CORR         <td>OFF (ON)                <td>Nuclear correction, either OFF, or ON (Soukup or Fluka or Gaussian fit)
## <tr><td>CUTOFF_BP_DEPTH      <td>1.05f                   <td>Depth cutoff as a percentage relative to the Bragg peak depth
## <tr><td>CUTOFF_CONV_SIGMA    <td>3.0f                    <td>Cutoff value in sigmas for the pre-convolution
## <tr><td>CUTOFF_KS_SIGMA      <td>3.0f                    <td>Cutoff value in sigmas for the kernel superposition
## <tr><td>CUTOFF_RAY_WEIGHT    <td>1.0f                    <td>Only trace rays with weight (particle number) above the cutoff
## <tr><td>IMG_DATA_DIRECTORY   <td>/tmp/                   <td>Directory where the DICOM image series are stored
## <tr><td>OUT_DATA_DIRECTORY   <td>/tmp/                   <td>Directory where the calculated results will be stored
## <tr><td>PHYS_DATA_DIRECTORY  <td>LUTs/                   <td>Directory where the LUTs are stored
## <tr><td>DEVICE_DEBUGGING     <td>OFF (ON)                <td>Debug code run on device
## <tr><td>EXECUTABLE_NAME      <td>RayTraceDicom           <td>Compiled binary name
## <tr><td>FAST_MATH            <td>ON (OFF)                <td>If use_fast_math is turned on/off
## <tr><td>REGISTERS_INFO       <td>OFF (ON)                <td>Outputs register per thread and memory information per kernel
## <tr><td>SEPARATE_COMPILATION <td>ON (OFF)                <td>Compile each source file separately. Results in less optimized code but good for debugging. OFF not working currently.
## <tr><td>ITK_DIR              <td>/usr/lib/cmake/ITK-4.10 <td>Path to itk shared library directory
## </table>
##
## Some parameters depend on each other, for example if you set release mode
## then the SEPARATE_COMPILATION is set to OFF. In debug mode, it is set to
## ON, and also DEVICE_DEBUGGING is activated. However, this cross-correlations
## only work if you run "cmake -D...". If you run ccmake, they are overriden.
## In relase mode, use-fast-math is activated by FindCUDA, CUDA_WRAP_SRCS
##
## The parameters that do not need to be changed are:
## <table>
## <caption id="other-cmake">Table of residual CMake parameters</caption>
## <tr><th>Variable                     <th>Value
## <tr><td>CMAKE_INSTALL_PREFIX         <td>/usr/local
## <tr><td>CUDA_HOST_COMPILER           <td>/usr/bin/cc
## <tr><td>CUDA_SDK_ROOT_DIR            <td>CUDA_SDK_ROOT_DIR-NOTFOUND
## <tr><td>CUDA_TOOLKIT_ROOT_DIR        <td>/usr/local/cuda
## <tr><td>CUDA_USE_STATIC_CUDA_RUNTIME <td>ON (OFF)
## <tr><td>CUDA_rt_LIBRARY              <td>usr/lib/x86_64-linux-gnu/librt.so
## </table>
##
## CMAKE_DOCUMENTATION_END
## See https://www.stack.nl/~dimitri/doxygen/manual/tables.html

cmake_minimum_required(VERSION 3.5)
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.2)
    message(FATAL_ERROR "GCC version must be at least 4.9.2!")
endif()

### Find packages
find_package(CUDA REQUIRED)
find_package(ITK REQUIRED)

### CUDA libraries
if(CUDA_FOUND)
    include_directories (
    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
    )
    #~ set(CUDA_LIBRARIES
        #~ ${CUDA_LIBRARIES}
        #~ ${CUDA_cusparse_LIBRARY}
    #~ )
else()
    message(FATAL_ERROR "Cannot find CUDA, did you set CUDA_TOOLKIT_ROOT_DIR?")
endif()

if(ITK_FOUND)
    include(${ITK_USE_FILE})
else()
    message(FATAL_ERROR "Cannot find ITK, did you set ITK_DIR?")
endif()

### LUT directory
set(PHYS_DATA_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/LUTs/ CACHE STRING "Directory where the LUTs are stored")
add_definitions(-DPHYS_DATA_DIRECTORY="${PHYS_DATA_DIRECTORY}")

### DICOM directory
set(IMG_DATA_DIRECTORY /tmp/ CACHE STRING "Directory where the DICOM image series are stored")
add_definitions(-DIMG_DATA_DIRECTORY="${IMG_DATA_DIRECTORY}")

### DICOM directory
set(OUT_DATA_DIRECTORY /tmp/ CACHE STRING "Directory where the output results are stored")
add_definitions(-DOUT_DATA_DIRECTORY="${OUT_DATA_DIRECTORY}")

### Code options
option(WATER_CUBE_TEST "If water cube test phantom is turned on/off" OFF)
if(WATER_CUBE_TEST)
  add_definitions(-DWATER_CUBE_TEST )
endif()
option(DOSE_TO_WATER "If dose is calculated in water" ON)
if(DOSE_TO_WATER)
  add_definitions(-DDOSE_TO_WATER )
endif()
option(FINE_GRAINED_TIMING "If fine grained timing is turned on/off" OFF)
if(FINE_GRAINED_TIMING)
  add_definitions(-DFINE_GRAINED_TIMING )
endif()
option(NOZZLE "If nozzle is included for sigma sq calculation" ON)
if(NOT NOZZLE)
  add_definitions(-DNO_NOZZLE )
endif()
option(FAST_MATH "If use_fast_math is turned on/off" ON)
if(FAST_MATH)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)
endif()

add_definitions(-DSOUKUP=0)    ## Soukup et al method
add_definitions(-DFLUKA=1)     ## Fluka method
add_definitions(-DGAUSS_FIT=2) ## Gaussian fit method

set(NUCLEAR_CORR "OFF" CACHE STRING "If nuclear correlation is calculated and how")
set_property(CACHE NUCLEAR_CORR PROPERTY STRINGS "SOUKUP" "FLUKA" "GAUSS_FIT" "OFF")
if(NUCLEAR_CORR STREQUAL "SOUKUP")
  add_definitions(-DNUCLEAR_CORR=SOUKUP )   ## Calculate NUCLEAR_CORR according to Soukup et al.
elseif(NUCLEAR_CORR STREQUAL "FLUKA")
  add_definitions(-DNUCLEAR_CORR=FLUKA )    ## Calculate NUCLEAR_CORR according to Fluka
elseif(NUCLEAR_CORR STREQUAL "GAUSS_FIT")
  add_definitions(-DNUCLEAR_CORR=GAUSS_FIT )## Calculate NUCLEAR_CORR according to Gaussian fit
endif()

set(CUTOFF_BP_DEPTH "1.05f" CACHE STRING "Depth cutoff as a percentage relative to the Bragg peak depth")
set(CUTOFF_CONV_SIGMA "3.0f" CACHE STRING "Cutoff value in sigmas for the pre-convolution")
set(CUTOFF_KS_SIGMA "3.0f" CACHE STRING "Cutoff value in sigmas for the kernel superposition")
set(CUTOFF_RAY_WEIGHT "1.0f" CACHE STRING "Only trace rays with weight (particle number) above the cutoff")

add_definitions(-DBP_DEPTH_CUTOFF=${CUTOFF_BP_DEPTH})
add_definitions(-DCONV_SIGMA_CUTOFF=${CUTOFF_CONV_SIGMA})
add_definitions(-DKS_SIGMA_CUTOFF=${CUTOFF_KS_SIGMA})
add_definitions(-DRAY_WEIGHT_CUTOFF=${CUTOFF_RAY_WEIGHT})

set(EXECUTABLE_NAME "RayTraceDicom" CACHE STRING "Compiled binary name")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14 -gencode arch=compute_35,code=sm_35 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --expt-relaxed-constexpr -rdc=true)#
set(CMAKE_CXX_STANDARD 14)

### Registers per thread in kernel info
option(REGISTERS_INFO "Outputs register per thread and memory information per kernel" OFF)
if (REGISTERS_INFO)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --ptxas-options=-v)
endif()

### Set release mode as default build type
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")# "MinSizeRel" "RelWithDebInfo"
endif()

### When in release mode, no separate compilation or device debugging
if (CMAKE_BUILD_TYPE STREQUAL Release)
    #~ set(SEPARATE_COMPILATION OFF CACHE BOOL "Compile each source file separately. Results in less optimized code but good for debugging.") DISABLE FOR THE MOMENT AS IT IS NOT WORKING
    set(DEVICE_DEBUGGING OFF CACHE BOOL "Debug device code.")
endif()

### When in debug mode: separate compilation and device debugging
if (CMAKE_BUILD_TYPE STREQUAL Debug)
    set(SEPARATE_COMPILATION ON CACHE BOOL "Compile each source file separately. Results in less optimized code but good for debugging.")
    set(DEVICE_DEBUGGING ON CACHE BOOL "Debug device code.")
endif()
if (DEVICE_DEBUGGING)
    set(CUDA_NVCC_FLAGS -G -g -Xcompiler -Wall ${CUDA_NVCC_FLAGS})#-rdynamic
endif()

include_directories(/home/huesof/Programs/RayTraceDicom/) # For QtCreator
#~ if (SEPARATE_COMPILATION)
set(CUDA_SEPARABLE_COMPILATION ON)

set(headers
    beam_settings.h
    constants.h
    cpu_convolution_1d.h
    cuda_errchk.cuh
    cuda_member.cuh
    density_and_sp_tracer_params.cuh
    dicom_reader.h
    energy_reader.h
    energy_struct.h
    fill_idd_and_sigma_params.cuh
    float3_affine_transform.cuh
    float3_from_fan_transform.cuh
    float3_idx_transform.cuh
    float3_to_fan_transform.cuh
    gpu_convolution_2d.cuh
    helper_float3.cuh
    host_image_3d.cuh
    kernel_wrapper.cuh
    matrix_3x3.cuh
    tracer_param_struct_div3.cuh
    tracer_param_struct3.h
    transfer_param_struct_div3.cuh
    vector_find.h
    vector_interpolate.h
    )
set(sources
    beam_settings.cpp
    cpu_convolution_1d.cpp
    cuda_errchk.cu
    density_and_sp_tracer_params.cu
    dicom_reader.cpp
    energy_reader.cpp
    fill_idd_and_sigma_params.cu
    float3_affine_transform.cu
    float3_from_fan_transform.cu
    float3_idx_transform.cu
    float3_to_fan_transform.cu
    gpu_convolution_2d.cu
    helper_float3.cu
    kernel_wrapper.cu
    main.cu
    matrix_3x3.cu
    tracer_param_struct_div3.cu
    transfer_param_struct_div3.cu
    )
cuda_add_executable(${EXECUTABLE_NAME} "${sources}" "${headers}")
#~ else()
    #~ add_definitions(-DALL_IN_ONE_COMPILATION )
    #~ set(CUDA_SEPARABLE_COMPILATION OFF)
    #~ cuda_add_executable(${EXECUTABLE_NAME} main.cu)
#~ endif()

### Add libraries
#cuda_add_cublas_to_target(${EXECUTABLE_NAME})
target_link_libraries(${EXECUTABLE_NAME} ${CUDA_LIBRARIES} ${ITK_LIBRARIES})
