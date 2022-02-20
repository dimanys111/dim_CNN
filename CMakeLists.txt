cmake_minimum_required(VERSION 3.5)

set(TARGET cnn)

project(${TARGET} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${TARGET} main.cpp)

find_package(PkgConfig REQUIRED)

set(LIBS_LIST mlpack)
foreach(LIB IN LISTS LIBS_LIST)
    pkg_check_modules(${LIB} REQUIRED ${LIB})
    target_link_libraries(${TARGET} ${${LIB}_LIBRARIES})
    target_include_directories(${TARGET} PUBLIC ${${LIB}_INCLUDE_DIRS})
    target_compile_options(${TARGET} PUBLIC ${${LIB}_CFLAGS_OTHER})
endforeach()


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


