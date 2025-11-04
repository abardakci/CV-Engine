cmake_minimum_required(VERSION 3.18)

project(VisionEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------
# Subprojects
# ----------------------------------------
add_subdirectory(modules/utils)          # libutils
add_subdirectory(modules/vision_toolkit) # libvision_toolkit
add_subdirectory(modules/allocator)      # liballoc
add_subdirectory(modules/dnn)            # libdnn
add_subdirectory(modules/vision)      # libdetection
add_subdirectory(modules/runtime)        # libruntime

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
# add_executable(sort_tracker app/track.cpp)

# target_link_libraries(sort_tracker PUBLIC
#     libruntime
# )

# ----------------------------------------
# Executable target: ESRGAN Upscaler
# ----------------------------------------
# add_executable(sort_tracker app/track.cpp)

# target_link_libraries(sort_tracker PUBLIC
#     libruntime
# )

# ----------------------------------------
# Install
# ----------------------------------------
include(GNUInstallDirs)

install(
    TARGETS 
        yolo
        # sort_tracker
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/config
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/models
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/assets
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})
