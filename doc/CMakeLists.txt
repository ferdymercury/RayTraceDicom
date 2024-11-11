## CMAKE_DOCUMENTATION_START CMakeLists.txt
##
## Main CMakeFile for compiling RayTraceDicom.
## Following variables can be configured when running ccmake:
## <table>
## <caption id="config-cmake">Table of configurable CMake parameters</caption>
## <tr><th>Variable             <th>Values                  <th>Description
## <tr><td>CMAKE_BUILD_TYPE     <td>Release (Debug)         <td>Choose the type of build
## <tr><td>COMPILE_SM20         <td>OFF (ON)                <td>Support Tesla C2075 GPUs
## <tr><td>COMPILE_SM35         <td>ON (OFF)                <td>Support Tesla K40c GPUs
## <tr><td>COMPILE_SM75         <td>OFF (ON)                <td>Support Turing GPUs
## <tr><td>COMPILE_SM90         <td>OFF (ON)                <td>Support Volta GPUs
## <tr><td>CUDA_SAMPLES_DIR     <td>samples/common/inc      <td>Path to headers of CUDA-samples
## <tr><td>WATER_CUBE_TEST      <td>OFF (ON)                <td>Use a water cube as test phantom, ignore IMG_DATA_DIRECTORY
## <tr><td>DOSE_TO_WATER        <td>ON (OFF)                <td>Calculate dose to water
## <tr><td>FINE_GRAINED_TIMING  <td>OFF (ON)                <td>Unset it to get more accurate total time, reports up to 30 ms longer execution time
## <tr><td>NOZZLE               <td>ON (OFF)                <td>Take into account air in the nozzle for sigma sq calculation
## <tr><td>NUCLEAR_CORR         <td>OFF (ON)                <td>Nuclear correction, either OFF, or ON (Soukup or Fluka or Gaussian fit)
## <tr><td>CUTOFF_BP_DEPTH      <td>1.05f                   <td>Depth cutoff as a percentage relative to the Bragg peak depth
## <tr><td>CUTOFF_CONV_SIGMA    <td>3.0f                    <td>Cutoff value in sigmas for the pre-convolution
## <tr><td>CUTOFF_KS_SIGMA      <td>3.0f                    <td>Cutoff value in sigmas for the kernel superposition
## <tr><td>CUTOFF_RAY_WEIGHT    <td>1.0f                    <td>Only trace rays with weight (particle number) above the cutoff
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

set (QHG_LOCATION qhelpgenerator CACHE STRING "Path to qhelpgenerator")

option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ON)
if(BUILD_DOCUMENTATION)
    find_package(Doxygen OPTIONAL_COMPONENTS dot)

    if(DOXYGEN_FOUND)
        set(DOXYGEN_PROJECT_NAME "RayTraceDicom")
        set(DOXYGEN_PROJECT_BRIEF "Sub-second pencil beam dose calculation on GPU for adaptive proton therapy")
        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_SOURCE_DIR}/README.md")
        set(DOXYGEN_WARN_NO_PARAMDOC YES)
        set(DOXYGEN_GENERATE_QHP YES)
        set(DOXYGEN_QCH_FILE "RayTraceDicom.qch")
        set(DOXYGEN_QHP_NAMESPACE org.doxygen.RTD.Project)
        set(DOXYGEN_QHG_LOCATION "${QHG_LOCATION}")
        set(DOXYGEN_GENERATE_LATEX NO)
        set(DOXYGEN_GENERATE_TREEVIEW YES)
        set(DOXYGEN_MACRO_EXPANSION YES)
        set(DOXYGEN_EXTENSION_MAPPING cu=c++ cuh=c++)
        set(DOXYGEN_FILE_PATTERNS *.h *.hpp *.cu *.cuh *.cpp)
        set(DOXYGEN_EXTRACT_PRIVATE YES)
        set(DOXYGEN_IMATE_PATH "../doc/")
        #set(DOXYGEN_HIDE_UNDOC_RELATIONS NO)
        #set(DOXYGEN_NUM_PROC_THREADS 0)
        #set(DOXYGEN_DOT_NUM_THREADS 1)
        set(DOXYGEN_DOT_MULTI_TARGETS YES)
        #set(DOXYGEN_UML_LOOK YES)
        #set(DOXYGEN_UML_LIMIT_NUM_FIELDS 50)
        #set(DOXYGEN_TEMPLATE_RELATIONS YES)
        set(DOXYGEN_DOT_IMAGE_FORMAT svg)
        set(DOXYGEN_INTERACTIVE_SVG YES)
        #set(DOXYGEN_DOT_GRAPH_MAX_NODES 100)
        #set(DOXYGEN_DOT_TRANSPARENT YES)
        #set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        doxygen_add_docs(doc ALL
          ${CMAKE_SOURCE_DIR}/README.md ${CMAKE_SOURCE_DIR}/doc/cmake.dox ${CMAKE_SOURCE_DIR}/src
          #ALLOW_DUPLICATE_CUSTOM_TARGETS
          #USE_STAMP_FILE
          COMMENT "Generating doxygen documentation for ${PROJECT_NAME}"
        )

        # install generated files
        install(
          DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html
          TYPE DOC
          OPTIONAL # because available only after "make doc"
        )
    else()
         message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    add_custom_target(dox ALL
      #DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    )
    include(make_documentation)
    #PARSE_CMAKE_DOCUMENTATION(INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" EXCLUDES "${CMAKE_CURRENT_BINARY_DIR}/*")
    #WRITE_CMAKE_DOCUMENTATION( "${CMAKE_CURRENT_SOURCE_DIR}/cmake.dox" SORTED )
endif()


# https://stackoverflow.com/questions/25376331/how-to-let-some-file-to-be-visiable-in-qtcreator
#set(DATA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../README.md)
