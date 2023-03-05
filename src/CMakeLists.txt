cmake_minimum_required(VERSION 3.24)
project(ANPR)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/include/opencv4/opencv2/")

set(CMAKE_CXX_STANDARD 17)
add_executable(ANPR main.cpp number_plate_recognition.cpp number_plate_recognition.h)
target_link_libraries(ANPR ${OpenCV_LIBS})