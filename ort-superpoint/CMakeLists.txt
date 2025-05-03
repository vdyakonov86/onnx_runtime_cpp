# find_package(OpenCV REQUIRED)

# add_library(yoloInference SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/yolo.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp)
# target_include_directories(yoloInference PRIVATE ${OpenCV_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include")
# target_link_libraries(yoloInference ${OpenCV_LIBRARIES})
target_sources(ort_utility
    PRIVATE SuperPoint.cpp
    PUBLIC FILE_SET HEADERS 
    BASE_DIRS ${PROJECT_SOURCE_DIR}
    FILES SuperPoint.hpp Utility.hpp)

