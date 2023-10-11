cmake_minimum_required(VERSION 3.16)
project(renderpy)
set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

MESSAGE(STATUS "CMake build type: ${CMAKE_BUILD_TYPE}")

# OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
link_directories(${OPENGL_LIBRARY_DIRS})

# Glog
add_subdirectory(thirdparty/glog-0.4.0)

# OpenCV
find_package(OpenCV REQUIRED)
MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Eigen
include_directories(thirdparty/eigen-3.4.0)

# JSON
include_directories(thirdparty/json/single_include)

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(pybind11)

# mLib
include_directories(thirdparty/mLib/include)
add_library(mLib
    thirdparty/mLib/include/mLibCore.cpp
)

# Thirdparty: glew, glm, eth3d-pipeline
add_subdirectory(thirdparty)
include_directories(
    thirdparty/glm-0.9.7.1/
	thirdparty/glew-1.13.0/include/
    thirdparty/eth3d-pipeline/src/
)

# Yaml
add_subdirectory(thirdparty/yaml-cpp-0.6.0)
include_directories(thirdparty/yaml-cpp-0.6.0/include)

include_directories(
    src
)

set(ALL_LIBS
    ${OpenCV_LIBS}
    GLEW_1130
    OpenGL::OpenGL
    OpenGL::EGL
    mLib
    glog::glog
    yaml-cpp
)

set(ALL_SRC
    src/opengl/egl.cpp
    src/opengl/opengl.cpp

    thirdparty/eth3d-pipeline/src/camera/camera_base.cc
    thirdparty/eth3d-pipeline/src/camera/camera_thin_prism.cc
    thirdparty/eth3d-pipeline/src/camera/camera_benchmark.cc
    thirdparty/eth3d-pipeline/src/camera/camera_fisheye_fov.cc
    thirdparty/eth3d-pipeline/src/camera/camera_polynomial_4.cc
    thirdparty/eth3d-pipeline/src/camera/camera_fisheye_polynomial_4.cc
    thirdparty/eth3d-pipeline/src/camera/camera_polynomial_tangential.cc
    thirdparty/eth3d-pipeline/src/camera/camera_fisheye_polynomial_tangential.cc
    thirdparty/eth3d-pipeline/src/camera/camera_pinhole.cc
    thirdparty/eth3d-pipeline/src/camera/camera_simple_pinhole.cc
    thirdparty/eth3d-pipeline/src/camera/camera_polynomial.cc
    thirdparty/eth3d-pipeline/src/camera/camera_radial.cc
    thirdparty/eth3d-pipeline/src/camera/camera_radial_fisheye.cc
    thirdparty/eth3d-pipeline/src/camera/camera_simple_radial.cc
    thirdparty/eth3d-pipeline/src/camera/camera_simple_radial_fisheye.cc
    thirdparty/eth3d-pipeline/src/camera/camera_full_opencv.cc
)

pybind11_add_module(renderpy src/pybind.cpp ${ALL_SRC})
target_link_libraries(renderpy PRIVATE ${ALL_LIBS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(renderpy
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
