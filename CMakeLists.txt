cmake_minimum_required(VERSION 3.18)

project(VisionEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------
# Find external packages
# ----------------------------------------
find_package(OpenCV REQUIRED)

# ----------------------------------------
# Subprojects
# ----------------------------------------
add_subdirectory(core/common)      # core_common
add_subdirectory(core/cv_wrapper)  # core_cv
add_subdirectory(core/algorithms)  # core_algorithms
add_subdirectory(core/trt_backend) # core_trt_backend
add_subdirectory(core/inference)   # core_infer
add_subdirectory(core/tracker)     # core_tracker
add_subdirectory(core/interfaces)  # core_interfaces
add_subdirectory(core/factory)

# ----------------------------------------
# Executable target: Object Detector
# ----------------------------------------
add_executable(yolo app/detect.cpp)

target_link_libraries(yolo PUBLIC
    ${OpenCV_LIBS}
    core_common
    core_interfaces
    core_cv
    core_algorithms
    core_factory
    core_infer
)

# ----------------------------------------
# Executable target: SORT Tracker
# ----------------------------------------
add_executable(sort_tracker app/track.cpp)

target_link_libraries(sort_tracker PUBLIC
    ${OpenCV_LIBS}
    core_common
    core_interfaces
    core_cv
    core_algorithms
    core_factory
    core_infer
    core_tracker
)

# ----------------------------------------
# Install
# ----------------------------------------
include(GNUInstallDirs)

install(TARGETS yolo sort_tracker
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/config
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/models
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/assets
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})
