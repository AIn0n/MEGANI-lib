cmake_minimum_required(VERSION 3.14)

project(
        pineapple-inteligence-lib
        VERSION 0.0.0
        DESCRIPTION "Minimal extensible neural networks implementation library for C language."
        HOMEPAGE_URL "https://github.com/AIn0n/pineapple-intelligence"
        LANGUAGES C
)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/sources/*.c)

add_library(pineapple_lib ${sources})
target_include_directories(pineapple_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(pineapple_lib PUBLIC c_std_99)

# link to math library if needed
if (NOT WIN32)
    target_link_libraries(pineapple_lib m)
endif ()

set_target_properties(pineapple_lib PROPERTIES OUTPUT_NAME "pineapple")
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

add_executable(pineapple_mnist_ann
        examples/mnist_ann.c
)

add_executable(pineapple_mnist_conv
        examples/mnist_conv.c
)

add_executable(pineapple_test
        sources/tests.c
        )

add_executable(pineapple_perf_test
        sources/perf_tests.c
        )

set(all_targets _mnist_ann pineapple_mnist_conv pineapple_test pineapple_perf_test)

foreach(target IN LISTS all_targets)
        target_link_libraries("${target}"
                PUBLIC pineapple_lib
        )
        MESSAGE(STATUS "here = ${target}")
endforeach()
