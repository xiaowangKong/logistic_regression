cmake_minimum_required(VERSION 3.6)
project(logistic_regression)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp LR.cpp LR.h)
add_executable(logistic_regression ${SOURCE_FILES})