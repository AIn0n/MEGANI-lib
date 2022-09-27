cmake_minimum_required(VERSION 3.14)

project(
        MEGANI-lib
        VERSION 0.0.0
        DESCRIPTION "Minimal Extensible GAN implementation library for C language."
        HOMEPAGE_URL "https://github.com/AIn0n/MEGANI-lib"
        LANGUAGES C
)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/sources/*.c)

add_library(MEGANI_lib ${sources})
target_include_directories(MEGANI_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(MEGANI_lib PUBLIC c_std_99)

# link to math library if needed
if (NOT WIN32)
    target_link_libraries(MEGANI_lib m)
endif ()

set_target_properties(MEGANI_lib PROPERTIES OUTPUT_NAME "MEGANI")
add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/sources/tests.c
        PRE_BUILD
        COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/generate_functional_tests.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )

add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/sources/perf_tests.c
        PRE_BUILD
        COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/mx_perf_test.py
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )

add_executable(MEGANI_mnist_ann
        examples/mnist_ann.c
)

add_executable(MEGANI_mnist_conv
        examples/mnist_conv.c
)

add_executable(MEGANI_test
        sources/tests.c
        )

add_executable(MEGANI_perf_test
        sources/perf_tests.c
        )


target_link_libraries(MEGANI_test
        MEGANI_lib
        )

target_link_libraries(MEGANI_perf_test
        MEGANI_lib
        )

find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(MEGANI_docs ALL
            COMMAND Doxygen::doxygen ${PROJECT_SOURCE_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating documentation with Doxygen"
            VERBATIM
            )
endif ()