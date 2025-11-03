file(GLOB SRC "src/*.cpp" "src/*.hpp")

add_library(libutils STATIC
    ${SRC}
)

target_include_directories(libutils 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
