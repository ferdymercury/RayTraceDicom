#~ if (SEPARATE_COMPILATION)
set(CUDA_SEPARABLE_COMPILATION ON)

set(headers
    beam_settings.h
    config.h
    constants.h
    #cpu_convolution_1d.h
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
    #tracer_param_struct_div3.cuh
    #tracer_param_struct3.h
    transfer_param_struct_div3.cuh
    vector_find.h
    vector_interpolate.h
    )
set(sources
    beam_settings.cpp
    config.cpp
    #cpu_convolution_1d.cpp
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
    #tracer_param_struct_div3.cu
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
target_link_libraries(${EXECUTABLE_NAME} ${CUDA_LIBRARIES} ${ITK_LIBRARIES} CLI11 dicom_interface)
