# 禁止根目录构建
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "禁止在源码目录中进行构建,防止编译生成文件污染源代码")
endif()

set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.15)

project(rmcs_auto_rune
        VERSION 0.1.0)

find_package(ament_cmake REQUIRED)
# cmake 函数/宏
include(cmake/VisCoreModule.cmake)

# cmake 编译选项
include(cmake/VisCoreCompilerOptions.cmake)


# VisCore 通用基础模块
add_subdirectory(common)

# VisCore 具体实现模块
add_subdirectory(modules)

# VisCore 测试模块
add_subdirectory(examples)

ament_export_include_directories(include)
ament_export_targets(${PROJECT_NAME}_targets HAS_LIBRARY_TARGET)

ament_package()
