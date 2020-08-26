cmake_minimum_required(VERSION 2.8)
project( DRD )
find_package( OpenCV REQUIRED )
add_executable( swt_modified SWT_modified.cpp )
target_link_libraries( swt_modified ${OpenCV_LIBS} )
set(CMAKE_BUILD_TYPE Debug)

