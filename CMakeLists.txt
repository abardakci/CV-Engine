cmake_minimum_required(VERSION 3.18)

project(VisionEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------
# Subprojects
# ----------------------------------------
add_subdirectory(core/core_utils) # core_utils
add_subdirectory(core/kalman_filter)  # libkalman_filter
add_subdirectory(core/vision_toolkit)  # libvision_toolkit
add_subdirectory(core/trt_backend) # libtrt_backend
add_subdirectory(core/detection)   # libdetection
add_subdirectory(core/trackers)     # libtrackers
add_subdirectory(core/interfaces)
add_subdirectory(core/factory)     # libfactory
add_subdirectory(core/runtime)     # libruntime

# ----------------------------------------
# Executable target: Object Detector
# ----------------------------------------
add_executable(yolo app/detect.cpp)

target_link_libraries(yolo PUBLIC
    libruntime
)

# ----------------------------------------
# Executable target: SORT Tracker
# ----------------------------------------
add_executable(sort_tracker app/track.cpp)

target_link_libraries(sort_tracker PUBLIC
    libruntime
)

# ----------------------------------------
# Install
# ----------------------------------------
include(GNUInstallDirs)

install(
    TARGETS 
        yolo
        sort_tracker
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/config
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/models
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/assets
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})
