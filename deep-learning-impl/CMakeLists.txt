cmake_minimum_required(VERSION 2.8.7)

project(twolayer)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug build")
ELSE()
    message("Release build")
ENDIF()
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)    
add_library(
    twolayer SHARED
    src/Affine.cpp
    src/Mymath.cpp
    src/Relu.cpp
    src/Sigmoid.cpp
    src/SoftmaxWithLoss.cpp
    #src/TwoLayerNet.cpp 
    src/Adam.cpp
    src/Dropout.cpp
    src/im2col.cpp
    src/col2im.cpp
    src/Convolution.cpp
    src/Pooling.cpp
    src/BatchNormalization.cpp
)

find_path(
    CBLASHEADER
    NAMES cblas.h
    PATHS /usr/include/x86_64-linux-gnu
    REQUIRED
)
find_library(
    CBLASLIB
    NAMES cblas
    HINTS /usr/lib/x86_64-linux-gnu
    REQUIRED
)
target_include_directories(    
    twolayer
    PUBLIC ${CBLASHEADER}
)
target_link_libraries(
    twolayer 
    ${CBLASLIB}  
)

