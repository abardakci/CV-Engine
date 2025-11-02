file(GLOB SRC "src/*.cpp")

add_library(libutils STATIC
    ${SRC}
)

target_include_directories(libutils 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
