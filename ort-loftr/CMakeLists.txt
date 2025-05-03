target_sources(ort_utility
    PRIVATE LoFTR.cpp
    PUBLIC FILE_SET HEADERS 
    BASE_DIRS ${PROJECT_SOURCE_DIR}
    FILES LoFTR.hpp)

