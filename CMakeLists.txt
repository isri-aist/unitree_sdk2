cmake_minimum_required(VERSION 3.5)
project(unitree_sdk2 VERSION 2.0.0)

## Project Options
option(BUILD_EXAMPLES "Build examples" ON)

## Set compiler to use c++ 17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

## Chosse build type
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

## Use GNUInstallDirs to install libraries into correct locations on all platforms.
include(GNUInstallDirs)

## Put all binary files into /bin and libraries into /lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

## Check system architecture
message(STATUS "Current system architecture: ${CMAKE_SYSTEM_PROCESSOR}")

# Import data tamer library
add_subdirectory(fmtlog)

## Import thirdparty libraries
add_subdirectory(thirdparty)

## Import Unitree SDK2 library
set(UNITREE_SDK_PATH ${CMAKE_CURRENT_LIST_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR})
find_library(UNITREE_SDK_LIB unitree_sdk2 PATHS ${UNITREE_SDK_PATH})

if (NOT UNITREE_SDK_LIB)
    message(FATAL_ERROR "Unitree SDK library for the architecture is not found")
else ()
    message(STATUS "Unitree SDK library found at: ${UNITREE_SDK_LIB}")
endif ()

message(STATUS "Importing: ${UNITREE_SDK_LIB}")

find_package(Threads REQUIRED)

add_library(unitree_sdk2 STATIC IMPORTED GLOBAL)
set_target_properties(unitree_sdk2 PROPERTIES
        IMPORTED_LOCATION ${UNITREE_SDK_LIB})
target_link_libraries(unitree_sdk2 INTERFACE ddsc ddscxx Threads::Threads)
target_include_directories(unitree_sdk2 INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

# Link fmt and libfort libraries
target_link_libraries(unitree_sdk2 INTERFACE fmt)
target_include_directories(unitree_sdk2 INTERFACE fmtlog)
target_include_directories(unitree_sdk2 INTERFACE libfort)
target_include_directories(unitree_sdk2 INTERFACE /home/h1user/paleziart/install/include)

#### ONNX RUNTIME

option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
include(FetchContent)

set(ONNXRUNTIME_ROOTDIR "/home/h1user/paleziart/onnxruntime")

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()

####

if (BUILD_EXAMPLES)
    add_subdirectory(example)
endif ()

## Install the library
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${UNITREE_SDK_LIB}
    DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES cmake/unitree_sdk2Targets.cmake
    DESTINATION lib/cmake/unitree_sdk2)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    unitree_sdk2ConfigVersion.cmake
    VERSION "${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH}"
    COMPATIBILITY ExactVersion)

configure_file(cmake/unitree_sdk2Config.cmake.in unitree_sdk2Config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/unitree_sdk2Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/unitree_sdk2ConfigVersion.cmake"
    DESTINATION lib/cmake/unitree_sdk2)
