target_sources(ort_utility
    PRIVATE LightGlue.cpp
    PUBLIC FILE_SET HEADERS 
    BASE_DIRS ${PROJECT_SOURCE_DIR}
    FILES LightGlue.hpp)