cmake_minimum_required(VERSION 3.5)

set(TARGET cnn)

project(${TARGET} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(${TARGET} main.cpp)

target_link_libraries(${TARGET} ${CONAN_LIBS})

find_package(PkgConfig REQUIRED)

set(LIBS_LIST mlpack)
foreach(LIB IN LISTS LIBS_LIST)
    pkg_check_modules(${LIB} REQUIRED ${LIB})
    target_link_libraries(${TARGET} ${${LIB}_LIBRARIES})
    target_include_directories(${TARGET} PUBLIC ${${LIB}_INCLUDE_DIRS})
    target_compile_options(${TARGET} PUBLIC ${${LIB}_CFLAGS_OTHER})
endforeach()



