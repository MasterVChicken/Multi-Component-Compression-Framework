cmake_minimum_required(VERSION 3.10)

project(ProgressiveCompression CXX C CUDA)

# Modify to CPP14 to support SZ3
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# ================================
# Configuration Options
# ================================
option(ENABLE_CUDA "Enable CUDA compressors." ON)
option(USE_EXTERNAL_DIR "Use external directory for libraries" OFF)

# Optional: Set external directory if needed
if(USE_EXTERNAL_DIR AND NOT EXTERNAL_DIR)
    set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external" CACHE PATH "External libraries directory")
endif()

# ================================
# Helper function to add library search paths
# ================================
function(add_library_search_paths lib_name)
    if(USE_EXTERNAL_DIR AND EXTERNAL_DIR)
        # Add external directory paths
        list(APPEND CMAKE_PREFIX_PATH 
            "${EXTERNAL_DIR}/${lib_name}/install"
            "${EXTERNAL_DIR}/${lib_name}/build"
            "${EXTERNAL_DIR}/${lib_name}")
        set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} PARENT_SCOPE)
    endif()
endfunction()

# ================================
# Find Libraries
# ================================

# Add search paths for external libraries if needed
if(USE_EXTERNAL_DIR)
    add_library_search_paths("SZ3")
    add_library_search_paths("MGARD")
    add_library_search_paths("zfp_cpu")
    add_library_search_paths("zfp_gpu")
    
    # Add common library paths
    list(APPEND CMAKE_PREFIX_PATH 
        "${EXTERNAL_DIR}/SZ3/install"
        "${EXTERNAL_DIR}/MGARD/install-cuda-ampere"
        "${EXTERNAL_DIR}/zfp_cpu/build"
        "${EXTERNAL_DIR}/zfp_gpu/build")
endif()

# Find SZ3
find_path(SZ3_INCLUDE_DIR
    NAMES SZ3/api/sz.hpp sz.hpp SZ/sz.hpp
    HINTS 
        ${SZ3_ROOT}/include
        $ENV{SZ3_ROOT}/include
    PATHS
        /usr/local/include
        /usr/include
        /opt/local/include
)

find_library(SZ3_LIBRARY
    NAMES SZ3c SZ3 sz3
    HINTS 
        ${SZ3_ROOT}/lib
        ${SZ3_ROOT}/lib64
        $ENV{SZ3_ROOT}/lib
        $ENV{SZ3_ROOT}/lib64
    PATHS
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /opt/local/lib
        /opt/local/lib64
)

# Create SZ3 imported target
if(SZ3_LIBRARY AND SZ3_INCLUDE_DIR)
    if(NOT TARGET SZ3::SZ3)
        if(SZ3_LIBRARY MATCHES "\\.so" OR SZ3_LIBRARY MATCHES "\\.dylib")
            add_library(SZ3::SZ3 SHARED IMPORTED)
        else()
            add_library(SZ3::SZ3 UNKNOWN IMPORTED)
        endif()
        
        set_target_properties(SZ3::SZ3 PROPERTIES
            IMPORTED_LOCATION "${SZ3_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${SZ3_INCLUDE_DIR}"
        )
    endif()
    message(STATUS "✓ Found SZ3: ${SZ3_LIBRARY}")
    message(STATUS "  SZ3 includes: ${SZ3_INCLUDE_DIR}")
else()
    message(WARNING "SZ3 not found. Set SZ3_ROOT or install SZ3 in standard locations.")
endif()

# Find MGARD
find_path(MGARD_INCLUDE_DIR
    NAMES mgard.hpp mgard/mgard.hpp
    HINTS 
        ${MGARD_ROOT}/include
        $ENV{MGARD_ROOT}/include
    PATHS
        /usr/local/include
        /usr/include
        /opt/local/include
)

find_library(MGARD_LIBRARY
    NAMES mgard
    HINTS 
        ${MGARD_ROOT}/lib
        ${MGARD_ROOT}/lib64
        $ENV{MGARD_ROOT}/lib
        $ENV{MGARD_ROOT}/lib64
    PATHS
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /opt/local/lib
        /opt/local/lib64
)

# Find MGARD dependencies
find_library(ZSTD_LIBRARY
    NAMES zstd
    HINTS 
        ${MGARD_ROOT}/lib
        ${MGARD_ROOT}/lib64
        $ENV{MGARD_ROOT}/lib
        $ENV{MGARD_ROOT}/lib64
    PATHS
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /opt/local/lib
        /opt/local/lib64
)

find_library(NVCOMP_LIBRARY
    NAMES nvcomp
    HINTS 
        ${MGARD_ROOT}/lib
        ${MGARD_ROOT}/lib64
        $ENV{MGARD_ROOT}/lib
        $ENV{MGARD_ROOT}/lib64
    PATHS
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /opt/local/lib
        /opt/local/lib64
)

# Create MGARD imported target
if(MGARD_LIBRARY AND MGARD_INCLUDE_DIR)
    if(NOT TARGET mgard::mgard)
        if(MGARD_LIBRARY MATCHES "\\.so" OR MGARD_LIBRARY MATCHES "\\.dylib")
            add_library(mgard::mgard SHARED IMPORTED)
        else()
            add_library(mgard::mgard UNKNOWN IMPORTED)
        endif()
        
        # Collect dependencies
        set(MGARD_DEPENDENCIES "")
        if(ZSTD_LIBRARY)
            list(APPEND MGARD_DEPENDENCIES "${ZSTD_LIBRARY}")
        endif()
        if(NVCOMP_LIBRARY)
            list(APPEND MGARD_DEPENDENCIES "${NVCOMP_LIBRARY}")
        endif()
        
        set_target_properties(mgard::mgard PROPERTIES
            IMPORTED_LOCATION "${MGARD_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MGARD_INCLUDE_DIR}"
        )
        
        if(MGARD_DEPENDENCIES)
            set_target_properties(mgard::mgard PROPERTIES
                INTERFACE_LINK_LIBRARIES "${MGARD_DEPENDENCIES}"
            )
        endif()
    endif()
    message(STATUS "✓ Found MGARD: ${MGARD_LIBRARY}")
    message(STATUS "  MGARD includes: ${MGARD_INCLUDE_DIR}")
    if(MGARD_DEPENDENCIES)
        message(STATUS "  MGARD dependencies: ${MGARD_DEPENDENCIES}")
    endif()
else()
    message(WARNING "MGARD not found. Set MGARD_ROOT or install MGARD in standard locations.")
endif()

# Find ZFP (try to find both CPU and GPU versions)
find_path(ZFP_INCLUDE_DIR
    NAMES zfp.h zfp/zfp.h
    HINTS 
        ${ZFP_ROOT}/include
        $ENV{ZFP_ROOT}/include
    PATHS
        /usr/local/include
        /usr/include
        /opt/local/include
)

# Find ZFP CPU library
find_library(ZFP_CPU_LIBRARY
    NAMES zfp
    HINTS 
        ${ZFP_CPU_ROOT}/lib
        ${ZFP_CPU_ROOT}/lib64
        $ENV{ZFP_CPU_ROOT}/lib
        $ENV{ZFP_CPU_ROOT}/lib64
        ${ZFP_ROOT}/lib
        ${ZFP_ROOT}/lib64
        $ENV{ZFP_ROOT}/lib
        $ENV{ZFP_ROOT}/lib64
    PATHS
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /opt/local/lib
        /opt/local/lib64
)

# Find ZFP GPU library (usually same as CPU, but can be separate build)
find_library(ZFP_GPU_LIBRARY
    NAMES zfp
    HINTS 
        ${ZFP_GPU_ROOT}/lib
        ${ZFP_GPU_ROOT}/lib64
        $ENV{ZFP_GPU_ROOT}/lib
        $ENV{ZFP_GPU_ROOT}/lib64
        ${ZFP_ROOT}/lib
        ${ZFP_ROOT}/lib64
        $ENV{ZFP_ROOT}/lib
        $ENV{ZFP_ROOT}/lib64
    PATHS
        /usr/local/lib
        /usr/local/lib64
        /usr/lib
        /usr/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /opt/local/lib
        /opt/local/lib64
)

# If no separate GPU library found, use CPU library
if(NOT ZFP_GPU_LIBRARY AND ZFP_CPU_LIBRARY)
    set(ZFP_GPU_LIBRARY ${ZFP_CPU_LIBRARY})
endif()

# Create ZFP CPU imported target
if(ZFP_CPU_LIBRARY AND ZFP_INCLUDE_DIR)
    if(NOT TARGET zfp::zfp_cpu)
        if(ZFP_CPU_LIBRARY MATCHES "\\.so" OR ZFP_CPU_LIBRARY MATCHES "\\.dylib")
            add_library(zfp::zfp_cpu SHARED IMPORTED)
        else()
            add_library(zfp::zfp_cpu UNKNOWN IMPORTED)
        endif()
        
        set_target_properties(zfp::zfp_cpu PROPERTIES
            IMPORTED_LOCATION "${ZFP_CPU_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ZFP_INCLUDE_DIR}"
        )
    endif()
    message(STATUS "✓ Found ZFP CPU: ${ZFP_CPU_LIBRARY}")
else()
    message(WARNING "ZFP CPU not found. Set ZFP_ROOT or ZFP_CPU_ROOT.")
endif()

# Create ZFP GPU imported target
if(ZFP_GPU_LIBRARY AND ZFP_INCLUDE_DIR)
    if(NOT TARGET zfp::zfp_gpu)
        if(ZFP_GPU_LIBRARY MATCHES "\\.so" OR ZFP_GPU_LIBRARY MATCHES "\\.dylib")
            add_library(zfp::zfp_gpu SHARED IMPORTED)
        else()
            add_library(zfp::zfp_gpu UNKNOWN IMPORTED)
        endif()
        
        set_target_properties(zfp::zfp_gpu PROPERTIES
            IMPORTED_LOCATION "${ZFP_GPU_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ZFP_INCLUDE_DIR}"
        )
    endif()
    message(STATUS "✓ Found ZFP GPU: ${ZFP_GPU_LIBRARY}")
else()
    message(WARNING "ZFP GPU not found. Set ZFP_ROOT or ZFP_GPU_ROOT.")
endif()

# ================================
# OpenMP
# ================================
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "✓ Found OpenMP: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found")
endif()

# ================================
# CPU Compressors
# ================================
add_library(cpu-compressors INTERFACE)
target_include_directories(cpu-compressors INTERFACE ${PROJECT_SOURCE_DIR}/include)

# Link available libraries
set(CPU_LIBS "")
if(TARGET OpenMP::OpenMP_CXX)
    list(APPEND CPU_LIBS OpenMP::OpenMP_CXX)
endif()
if(TARGET zfp::zfp_cpu)
    list(APPEND CPU_LIBS zfp::zfp_cpu)
endif()
if(TARGET SZ3::SZ3)
    list(APPEND CPU_LIBS SZ3::SZ3)
endif()
if(TARGET mgard::mgard)
    list(APPEND CPU_LIBS mgard::mgard)
endif()

if(CPU_LIBS)
    target_link_libraries(cpu-compressors INTERFACE ${CPU_LIBS})
    message(STATUS "CPU compressors will link: ${CPU_LIBS}")
else()
    message(WARNING "No compression libraries found for CPU compressors")
endif()

# Build CPU executable if source exists
if(EXISTS "${CMAKE_SOURCE_DIR}/testCPUCompressors.cpp")
    add_executable(ProgressiveCompressionCPU testCPUCompressors.cpp)
    target_link_libraries(ProgressiveCompressionCPU cpu-compressors)

    target_compile_options(ProgressiveCompressionCPU PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fopenmp>
        $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    )
    message(STATUS "✓ Will build ProgressiveCompressionCPU executable")
endif()

# ================================
# GPU Compressors
# ================================
if(ENABLE_CUDA)
    enable_language(CUDA)
    
    # Set CUDA architecture to 87 for better compatibility
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "87")
        message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # GPU Compressors library
    set(gpu_compressors_src
        include/GPUMGARDCompressor.hpp
        include/GPUZFPCompressor.hpp
    )
    set_source_files_properties(${gpu_compressors_src} PROPERTIES LANGUAGE CUDA)

    add_library(gpu-compressors ${gpu_compressors_src})
    target_include_directories(gpu-compressors PUBLIC ${PROJECT_SOURCE_DIR}/include)
    
    # Link available GPU libraries
    set(GPU_LIBS "")
    if(TARGET mgard::mgard)
        list(APPEND GPU_LIBS mgard::mgard)
    endif()
    if(TARGET zfp::zfp_gpu)
        list(APPEND GPU_LIBS zfp::zfp_gpu)
    endif()
    
    if(GPU_LIBS)
        target_link_libraries(gpu-compressors PUBLIC ${GPU_LIBS})
        message(STATUS "GPU compressors will link: ${GPU_LIBS}")
    else()
        message(WARNING "No GPU compression libraries found")
    endif()
    
    set_target_properties(gpu-compressors PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_compile_options(gpu-compressors PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
        $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
    )

    # GPU MGARD executable
    if(EXISTS "${CMAKE_SOURCE_DIR}/testMGARDCompressor.cpp" AND TARGET mgard::mgard)
        set_source_files_properties(testMGARDCompressor.cpp PROPERTIES LANGUAGE CUDA)
        add_executable(ProgressiveCompressionGPUMGARD testMGARDCompressor.cpp)
        target_link_libraries(ProgressiveCompressionGPUMGARD gpu-compressors)
        target_include_directories(ProgressiveCompressionGPUMGARD PRIVATE ${PROJECT_SOURCE_DIR}/include)
        
        target_compile_options(ProgressiveCompressionGPUMGARD PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
            $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
        )
        message(STATUS "✓ Will build ProgressiveCompressionGPUMGARD executable")
    endif()

    # GPU ZFP executable
    if(EXISTS "${CMAKE_SOURCE_DIR}/testZFPCompressor.cpp" AND TARGET zfp::zfp_gpu)
        set_source_files_properties(testZFPCompressor.cpp PROPERTIES LANGUAGE CUDA)
        add_executable(ProgressiveCompressionGPUZFP testZFPCompressor.cpp)
        target_link_libraries(ProgressiveCompressionGPUZFP gpu-compressors)
        target_include_directories(ProgressiveCompressionGPUZFP PRIVATE ${PROJECT_SOURCE_DIR}/include)
        
        target_compile_options(ProgressiveCompressionGPUZFP PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -fPIC>
            $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
        )
        message(STATUS "✓ Will build ProgressiveCompressionGPUZFP executable")
    endif()
else()
    message(STATUS "CUDA disabled - GPU compressors will not be built")
endif()

# ================================
# Configuration Summary
# ================================
message(STATUS "=== Configuration Summary ===")
message(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "USE_EXTERNAL_DIR: ${USE_EXTERNAL_DIR}")
if(USE_EXTERNAL_DIR)
    message(STATUS "External Dir: ${EXTERNAL_DIR}")
endif()
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
if(ENABLE_CUDA)
    message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Libraries found:")
if(SZ3_LIBRARY)
    message(STATUS "  SZ3: ${SZ3_LIBRARY}")
endif()
if(MGARD_LIBRARY)
    message(STATUS "  MGARD: ${MGARD_LIBRARY}")
endif()
if(ZFP_CPU_LIBRARY)
    message(STATUS "  ZFP CPU: ${ZFP_CPU_LIBRARY}")
endif()
if(ZFP_GPU_LIBRARY)
    message(STATUS "  ZFP GPU: ${ZFP_GPU_LIBRARY}")
endif()
message(STATUS "=============================")