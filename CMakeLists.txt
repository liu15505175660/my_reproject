cmake_minimum_required(VERSION 3.0)
project(my_reprojection)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置编译类型为 Debug
set(CMAKE_BUILD_TYPE Debug)

# 查找 OpenCV
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找 PCL
find_package(PCL 1.10 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 设置可执行文件
add_executable(my_reprojection reprojection.cpp)

# 链接 OpenCV 和 PCL 库
target_link_libraries(my_reprojection ${OpenCV_LIBS} ${PCL_LIBRARIES})

