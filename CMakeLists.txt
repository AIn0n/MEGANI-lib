cmake_minimum_required(VERSION 3.14)

project(
        MEGANI-lib
        VERSION 0.0.0
        DESCRIPTION "Minimal Extensible GAN implementation library for C language."
        HOMEPAGE_URL "https://github.com/AIn0n/MEGANI-lib"
        LANGUAGES C
)

set(TEST_GENERATOR "generate_functional_tests.py" CACHE PATH "python script used to generate tests")

file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/sources/*.c)

add_library(MEGANI_lib ${sources})
target_include_directories(MEGANI_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(MEGANI_lib PUBLIC c_std_99)

# link to math library if needed
if (NOT WIN32)
    target_link_libraries(MEGANI_lib m)
endif ()

add_custom_command(OUTPUT ${PROJECT_SOURCE_DIR}/sources/main.c
                   PRE_BUILD
                   COMMAND python3 ${PROJECT_SOURCE_DIR}/scripts/${TEST_GENERATOR}
                   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_executable(MEGANI_test
    sources/main.c
)


target_link_libraries(MEGANI_test
    MEGANI_lib
)
