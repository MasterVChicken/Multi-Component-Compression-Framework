cmake_minimum_required(VERSION 3.10)

project(ProgressiveCompression CXX C CUDA)

# Modify to CPP14 to support SZ3
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Hard code all external lib path
set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")

# SZ3
set(SZ3_INSTALL_DIR "${EXTERNAL_DIR}/SZ3/install")
set(SZ3_LIB_DIR "${SZ3_INSTALL_DIR}/lib64")  
set(SZ3_INCLUDE_DIR "${SZ3_INSTALL_DIR}/include")

# MGARD
set(MGARD_INSTALL_DIR "${EXTERNAL_DIR}/MGARD/install-cuda-ampere")
set(MGARD_LIB_DIR "${MGARD_INSTALL_DIR}/lib64") 
set(MGARD_LIB_DIR_DEPS "${MGARD_INSTALL_DIR}/lib")  
set(MGARD_INCLUDE_DIR "${MGARD_INSTALL_DIR}/include")

# ZFP CPU
set(ZFP_CPU_BUILD_DIR "${EXTERNAL_DIR}/zfp_cpu/build")
set(ZFP_CPU_LIB_DIR "${ZFP_CPU_BUILD_DIR}/lib64")  
set(ZFP_CPU_INCLUDE_DIR "${EXTERNAL_DIR}/zfp_cpu/include")

# ZFP GPU
set(ZFP_GPU_BUILD_DIR "${EXTERNAL_DIR}/zfp_gpu/build")
set(ZFP_GPU_LIB_DIR "${ZFP_GPU_BUILD_DIR}/lib64")  
set(ZFP_GPU_INCLUDE_DIR "${EXTERNAL_DIR}/zfp_gpu/include")

# Debug info
message(STATUS "=== Debugging Library Files ===")
file(GLOB SZ3_FILES "${SZ3_LIB_DIR}/*")
message(STATUS "SZ3 lib64 directory contents: ${SZ3_FILES}")

file(GLOB MGARD_FILES "${MGARD_LIB_DIR}/*")
message(STATUS "MGARD lib64 directory contents: ${MGARD_FILES}")

file(GLOB MGARD_DEP_FILES "${MGARD_LIB_DIR_DEPS}/*")
message(STATUS "MGARD lib dependency directory contents: ${MGARD_DEP_FILES}")

file(GLOB ZFP_CPU_FILES "${ZFP_CPU_LIB_DIR}/*")
message(STATUS "ZFP CPU lib64 directory contents: ${ZFP_CPU_FILES}")

file(GLOB ZFP_GPU_FILES "${ZFP_GPU_LIB_DIR}/*")
message(STATUS "ZFP GPU lib64 directory contents: ${ZFP_GPU_FILES}")

# Find library

find_library(SZ3_LIBRARY
    NAMES 
        libSZ3c.so SZ3c.so
        libSZ3.so SZ3.so 
        libsz3.so sz3.so
        libSZ3c.a SZ3c.a
        libSZ3.a SZ3.a 
        libsz3.a sz3.a
        SZ3c SZ3 sz3
    PATHS ${SZ3_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(MGARD_LIBRARY
    NAMES 
        libmgard.so mgard.so
        libmgard.a mgard.a
        mgard
    PATHS ${MGARD_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(ZSTD_LIBRARY
    NAMES 
        libzstd.so zstd.so
        libzstd.a zstd.a
        zstd
    PATHS ${MGARD_LIB_DIR_DEPS}
    NO_DEFAULT_PATH
)

find_library(NVCOMP_LIBRARY
    NAMES 
        libnvcomp.so nvcomp.so
        libnvcomp.a nvcomp.a
        nvcomp
    PATHS ${MGARD_LIB_DIR_DEPS}
    NO_DEFAULT_PATH
)

find_library(ZFP_CPU_LIBRARY
    NAMES 
        libzfp.so zfp.so
        libzfp.a zfp.a
        zfp
    PATHS ${ZFP_CPU_LIB_DIR}
    NO_DEFAULT_PATH
)

find_library(ZFP_GPU_LIBRARY
    NAMES 
        libzfp.so zfp.so
        libzfp.a zfp.a
        zfp
    PATHS ${ZFP_GPU_LIB_DIR}
    NO_DEFAULT_PATH
)


# SZ3
if(SZ3_LIBRARY AND EXISTS ${SZ3_INCLUDE_DIR})
    if(SZ3_LIBRARY MATCHES "\\.so")
        add_library(SZ3::SZ3 SHARED IMPORTED)
    else()
        add_library(SZ3::SZ3 UNKNOWN IMPORTED)
    endif()
    
    set_target_properties(SZ3::SZ3 PROPERTIES
        IMPORTED_LOCATION "${SZ3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SZ3_INCLUDE_DIR}"
    )
    message(STATUS "✓ Found SZ3: ${SZ3_LIBRARY}")
else()
    message(FATAL_ERROR "✗ SZ3 not found. Library: ${SZ3_LIBRARY}, Include: ${SZ3_INCLUDE_DIR}")
endif()

# MGARD
if(MGARD_LIBRARY AND EXISTS ${MGARD_INCLUDE_DIR})
    if(MGARD_LIBRARY MATCHES "\\.so")
        add_library(mgard::mgard SHARED IMPORTED)
    else()
        add_library(mgard::mgard UNKNOWN IMPORTED)
    endif()
    
    # Collect all dependencies
    set(MGARD_DEPENDENCIES "")
    
    if(ZSTD_LIBRARY)
        list(APPEND MGARD_DEPENDENCIES "${ZSTD_LIBRARY}")
        message(STATUS "✓ Found ZSTD dependency: ${ZSTD_LIBRARY}")
    endif()
    
    if(NVCOMP_LIBRARY)
        list(APPEND MGARD_DEPENDENCIES "${NVCOMP_LIBRARY}")
        message(STATUS "✓ Found NVCOMP dependency: ${NVCOMP_LIBRARY}")
    endif()
    
    set_target_properties(mgard::mgard PROPERTIES
        IMPORTED_LOCATION "${MGARD_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${MGARD_INCLUDE_DIR}"
    )
    
    # Add dependencies if possible
    if(MGARD_DEPENDENCIES)
        set_target_properties(mgard::mgard PROPERTIES
            INTERFACE_LINK_LIBRARIES "${MGARD_DEPENDENCIES}"
        )
    endif()
    
    message(STATUS "✓ Found MGARD: ${MGARD_LIBRARY}")
    if(MGARD_DEPENDENCIES)
        message(STATUS "  MGARD dependencies: ${MGARD_DEPENDENCIES}")
    endif()
else()
    message(FATAL_ERROR "✗ MGARD not found. Library: ${MGARD_LIBRARY}, Include: ${MGARD_INCLUDE_DIR}")
endif()

# ZFP CPU
if(ZFP_CPU_LIBRARY AND EXISTS ${ZFP_CPU_INCLUDE_DIR})
    if(ZFP_CPU_LIBRARY MATCHES "\\.so")
        add_library(zfp::zfp_cpu SHARED IMPORTED)
    else()
        add_library(zfp::zfp_cpu UNKNOWN IMPORTED)
    endif()
    
    set_target_properties(zfp::zfp_cpu PROPERTIES
        IMPORTED_LOCATION "${ZFP_CPU_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZFP_CPU_INCLUDE_DIR}"
    )
    message(STATUS "✓ Found ZFP CPU: ${ZFP_CPU_LIBRARY}")
else()
    message(FATAL_ERROR "✗ ZFP CPU not found. Library: ${ZFP_CPU_LIBRARY}, Include: ${ZFP_CPU_INCLUDE_DIR}")
endif()

# ZFP GPU
if(ZFP_GPU_LIBRARY AND EXISTS ${ZFP_GPU_INCLUDE_DIR})
    if(ZFP_GPU_LIBRARY MATCHES "\\.so")
        add_library(zfp::zfp_gpu SHARED IMPORTED)
    else()
        add_library(zfp::zfp_gpu UNKNOWN IMPORTED)
    endif()
    
    set_target_properties(zfp::zfp_gpu PROPERTIES
        IMPORTED_LOCATION "${ZFP_GPU_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZFP_GPU_INCLUDE_DIR}"
    )
    message(STATUS "✓ Found ZFP GPU: ${ZFP_GPU_LIBRARY}")
else()
    message(FATAL_ERROR "✗ ZFP GPU not found. Library: ${ZFP_GPU_LIBRARY}, Include: ${ZFP_GPU_INCLUDE_DIR}")
endif()

# ================================
# OpenMP
# ================================
find_package(OpenMP REQUIRED)

# ================================
# CPU Compressors
# ================================
option(ENABLE_CUDA "Enable CUDA compressors." ON)

add_library(cpu-compressors INTERFACE)
target_include_directories(cpu-compressors INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cpu-compressors INTERFACE 
    zfp::zfp_cpu 
    SZ3::SZ3 
    mgard::mgard
    OpenMP::OpenMP_CXX
)

if(EXISTS "${CMAKE_SOURCE_DIR}/testCPUCompressors.cpp")
    add_executable(ProgressiveCompressionCPU testCPUCompressors.cpp)
    target_link_libraries(ProgressiveCompressionCPU cpu-compressors)

    target_compile_options(ProgressiveCompressionCPU PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fopenmp>
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    )
endif()

# ================================
# GPU Compressors
# ================================
if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 90)
    
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # GPU Compressors LIB
    set(gpu_compressors_src
        include/GPUMGARDCompressor.hpp
        include/GPUZFPCompressor.hpp
    )
    set_source_files_properties(${gpu_compressors_src} PROPERTIES LANGUAGE CUDA)

    add_library(gpu-compressors ${gpu_compressors_src})
    target_link_libraries(gpu-compressors PUBLIC 
        mgard::mgard 
        zfp::zfp_gpu
    )
    target_include_directories(gpu-compressors PUBLIC ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(gpu-compressors PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_compile_options(gpu-compressors PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
        $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
    )

    # GPU MGARD EXEC
    if(EXISTS "${CMAKE_SOURCE_DIR}/testMGARDCompressor.cpp")
        set_source_files_properties(testMGARDCompressor.cpp PROPERTIES LANGUAGE CUDA)
        add_executable(ProgressiveCompressionGPUMGARD testMGARDCompressor.cpp)
        target_link_libraries(ProgressiveCompressionGPUMGARD gpu-compressors)
        target_include_directories(ProgressiveCompressionGPUMGARD PRIVATE ${PROJECT_SOURCE_DIR}/include)
        
        target_compile_options(ProgressiveCompressionGPUMGARD PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
            $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
        )
    endif()

    # GPU ZFP EXEC
    if(EXISTS "${CMAKE_SOURCE_DIR}/testZFPCompressor.cpp")
        set_source_files_properties(testZFPCompressor.cpp PROPERTIES LANGUAGE CUDA)
        add_executable(ProgressiveCompressionGPUZFP testZFPCompressor.cpp)
        target_link_libraries(ProgressiveCompressionGPUZFP gpu-compressors)
        target_include_directories(ProgressiveCompressionGPUZFP PRIVATE ${PROJECT_SOURCE_DIR}/include)
        
        target_compile_options(ProgressiveCompressionGPUZFP PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
            $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
        )
    endif()
else()
    message(STATUS "CUDA disabled - GPU compressors will not be built")
endif()

message(STATUS "=== Configuration Summary ===")
message(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "External Dir: ${EXTERNAL_DIR}")
message(STATUS "SZ3: ${SZ3_LIBRARY}")
message(STATUS "MGARD: ${MGARD_LIBRARY}")
message(STATUS "ZFP CPU: ${ZFP_CPU_LIBRARY}")
message(STATUS "ZFP GPU: ${ZFP_GPU_LIBRARY}")
message(STATUS "=============================")