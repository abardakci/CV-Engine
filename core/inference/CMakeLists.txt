add_library(core_infer STATIC
    trt_engine.cpp
    yolo_detector.cpp
)

target_include_directories(core_infer PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(core_infer PUBLIC
    ${OpenCV_LIBS}
    core_common
    core_algo
)
