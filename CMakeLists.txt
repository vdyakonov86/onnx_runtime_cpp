cmake_minimum_required(VERSION 3.25)

project(ort_utility VERSION 0.0.2)
set (CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(onnxruntime_INSTALL_PREFIX /onnx_runtime_cpp/onnxruntime)
set(onnxruntime_INCLUDE_DIRS
  ${onnxruntime_INSTALL_PREFIX}/include
  ${onnxruntime_INSTALL_PREFIX}/include/core/providers
)

find_library(onnxruntime_LIBS NAMES onnxruntime PATHS ${onnxruntime_INSTALL_PREFIX}/lib)
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET)

if(CUDA_FOUND AND USE_GPU)
  add_definitions(-DENABLE_GPU=1)
  include(tensorrt)
  if (TENSORRT_FOUND)
    add_definitions(-DENABLE_TENSORRT=1)
  endif()
else()
  add_definitions(-DENABLE_GPU=0)
  add_definitions(-DENABLE_TENSORRT=0)
endif()


add_compile_options(
  "$<$<CONFIG:Debug>:-DENABLE_DEBUG=1>"
  "$<$<CONFIG:Release>:-DENABLE_DEBUG=0>"
)

add_subdirectory(src)

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif(BUILD_EXAMPLES)

add_subdirectory(ort-superpoint)

# For installing package
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    FILE_SET HEADERS
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)

install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "ort_utilityConfigVersion.cmake"
    VERSION ${ort_utility_VERSION}
    COMPATIBILITY AnyNewerVersion)

install(EXPORT ort_utilityTargets
    FILE ort_utilityTargets.cmake
    NAMESPACE ort_utility::
    DESTINATION lib/cmake/ort_utility)

install(FILES "ort_utilityConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/ort_utilityConfigVersion.cmake"
    DESTINATION lib/cmake/ort_utility)

# uninstall target
if(NOT TARGET uninstall)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)

add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()