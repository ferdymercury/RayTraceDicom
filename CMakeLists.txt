

cmake_minimum_required(VERSION 3.5)
project(RayTraceDicom)
if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.2)
    message(FATAL_ERROR "GCC version must be at least 4.9.2!")
endif()

### Find packages
find_package(CUDA REQUIRED)
find_package(ITK REQUIRED)

### CUDA libraries
if(CUDA_FOUND)
    set(CUDA_SAMPLES_DIR ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc CACHE STRING "Directory where CUDA samples headers are stored")
    include_directories (
    ${CUDA_SAMPLES_DIR}
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
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES --expt-relaxed-constexpr -rdc=true)
if (COMPILE_SM20)
    set(CMAKE_CXX_STANDARD 11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11)
else()
    set(CMAKE_CXX_STANDARD 14)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14)
endif()


### Registers per thread in kernel info
option(REGISTERS_INFO "Outputs register per thread and memory information per kernel" OFF)
if (REGISTERS_INFO)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --ptxas-options=-v)
endif()

### Tesla C2075 GPUs
set (COMPILE_SM20 OFF CACHE BOOL "Support Tesla C2075 GPUs")# Need to install Cuda 8 instead of Cuda 9
if (COMPILE_SM20)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20)
endif()

### Tesla K40c GPUs
set (COMPILE_SM35 ON CACHE BOOL "Support Tesla K40c GPUs")
if (COMPILE_SM35)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35)
endif()

### Turing GPUs
set (COMPILE_SM75 OFF CACHE BOOL "Support Turing GPUs")
if (COMPILE_SM75)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75)
endif()

### Volta GPUs
set (COMPILE_SM90 OFF CACHE BOOL "Support Volta GPUs")
if (COMPILE_SM90)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_90,code=sm_90)
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

add_subdirectory(extern/CLI11)
add_subdirectory(extern/dicom-interface)

add_subdirectory(src)

### cmake-modules
if(NOT DEFINED CMAKEMODULES_DIR)
set (CMAKEMODULES_DIR "/opt/cmake-modules" CACHE STRING "rpavlik cmake-modules git repository")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKEMODULES_DIR}")

add_subdirectory(doc)#Later to know all the headers
