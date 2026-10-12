cmake_minimum_required(VERSION 3.20)
project(axman VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Options and Build Configurations
# -----------------------------------------------------------------------------
option(AXMAN_BUILD_SHARED "Build dynamic shared library (libaxman.so / axman.dll)" ON)
option(AXMAN_BUILD_STATIC "Build static archive library (libaxman.a / axman.lib)" ON)
option(AXMAN_BUILD_TESTS "Build test suite executables" ON)
option(AXMAN_ENABLE_CUDA "Enable CUDA accelerator kernels" OFF)
option(AXMAN_USE_LIBTORCH "Link against LibTorch runtime if available" OFF)

# -----------------------------------------------------------------------------
# Compiler Flags and Export Visibility
# -----------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /EHsc /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# -----------------------------------------------------------------------------
# Core Source Files & Includes
# -----------------------------------------------------------------------------
set(AXMAN_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/core/runtime
)

set(AXMAN_SOURCES
    src/memory.c
    src/currying.c
    src/hash.c
    core/runtime/axman_engine.cpp
)

set(AXMAN_HEADERS
    include/axman_c_api.h
    include/axman_currying.h
    include/axman_hash.h
    core/runtime/axman_engine_internal.hpp
)

# Optional LibTorch Integration
if(AXMAN_USE_LIBTORCH)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        message(STATUS "LibTorch found: ${TORCH_LIBRARIES}")
        include_directories(${TORCH_INCLUDE_DIRS})
        add_compile_definitions(AXMAN_HAS_LIBTORCH=1)
    else()
        message(WARNING "LibTorch requested but not found. Falling back to native C++20 engine.")
    endif()
endif()

# Optional CUDA Integration
if(AXMAN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_compile_definitions(AXMAN_HAS_CUDA=1)
    else()
        message(WARNING "CUDA requested but no CUDA compiler found. Building CPU-only.")
    endif()
endif()

# -----------------------------------------------------------------------------
# Targets: Dynamic and Static Libraries
# -----------------------------------------------------------------------------

# Shared library (libaxman.so / axman.dll)
if(AXMAN_BUILD_SHARED)
    add_library(axman_shared SHARED ${AXMAN_SOURCES} ${AXMAN_HEADERS})
    set_target_properties(axman_shared PROPERTIES
        OUTPUT_NAME "axman"
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        C_VISIBILITY_PRESET "default"
        CXX_VISIBILITY_PRESET "default"
        VISIBILITY_INLINES_HIDDEN OFF
    )
    target_compile_definitions(axman_shared PRIVATE AXMAN_EXPORTS=1 AXMAN_BUILDING_DLL=1)
    target_include_directories(axman_shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE ${AXMAN_INCLUDES}
    )
    if(WIN32)
        # Ensure standard dll naming on Windows: axman.dll / libaxman.dll.a
        set_target_properties(axman_shared PROPERTIES PREFIX "lib")
    endif()
    if(Torch_FOUND)
        target_link_libraries(axman_shared PRIVATE ${TORCH_LIBRARIES})
    endif()
endif()

# Static library (libaxman.a / axman.lib)
if(AXMAN_BUILD_STATIC)
    add_library(axman_static STATIC ${AXMAN_SOURCES} ${AXMAN_HEADERS})
    set_target_properties(axman_static PROPERTIES
        OUTPUT_NAME "axman_static"
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_definitions(axman_static PUBLIC AXMAN_STATIC=1 AXMAN_STATIC_LIB=1)
    target_include_directories(axman_static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE ${AXMAN_INCLUDES}
    )
    if(Torch_FOUND)
        target_link_libraries(axman_static PRIVATE ${TORCH_LIBRARIES})
    endif()
endif()

# -----------------------------------------------------------------------------
# Test Executables
# -----------------------------------------------------------------------------
if(AXMAN_BUILD_TESTS)
    enable_testing()

    # 1. Phase 0 C-tests verification executable
    add_executable(run_phase0_tests tests/run_all_phase0_tests.c ${AXMAN_SOURCES})
    target_compile_definitions(run_phase0_tests PRIVATE AXMAN_STATIC=1)
    target_include_directories(run_phase0_tests PRIVATE ${AXMAN_INCLUDES})
    add_test(NAME Phase0_Verification COMMAND run_phase0_tests)

    # 2. Phase 3 Engine unit tests (dynamically linked against axman_shared if built)
    if(AXMAN_BUILD_SHARED)
        add_executable(test_engine_shared tests/test_engine.cpp)
        target_include_directories(test_engine_shared PRIVATE ${AXMAN_INCLUDES})
        target_link_libraries(test_engine_shared PRIVATE axman_shared)
        add_test(NAME Engine_Dynamic_Link_Test COMMAND test_engine_shared)
    endif()

    # 3. Phase 3 Engine unit tests (statically linked)
    if(AXMAN_BUILD_STATIC)
        add_executable(test_engine_static tests/test_engine.cpp)
        target_include_directories(test_engine_static PRIVATE ${AXMAN_INCLUDES})
        target_link_libraries(test_engine_static PRIVATE axman_static)
        add_test(NAME Engine_Static_Link_Test COMMAND test_engine_static)
    endif()

    # 4. Phase 3.3 Safetensors archive parser and loading test
    if(AXMAN_BUILD_SHARED)
        add_executable(test_safetensors tests/test_safetensors.cpp)
        target_include_directories(test_safetensors PRIVATE ${AXMAN_INCLUDES})
        target_link_libraries(test_safetensors PRIVATE axman_shared)
        add_test(NAME Safetensors_Archive_Test COMMAND test_safetensors)
    endif()

    # 5. Phase 4.2 Modern C++20 axman-cpp wrapper test
    if(AXMAN_BUILD_SHARED)
        add_executable(test_cpp_bindings bindings/cpp/tests/test_cpp_bindings.cpp)
        target_include_directories(test_cpp_bindings PRIVATE ${AXMAN_INCLUDES} bindings/cpp/include)
        target_link_libraries(test_cpp_bindings PRIVATE axman_shared)
        add_test(NAME Cpp20_Bindings_Test COMMAND test_cpp_bindings)

        # 6. Phase 4.5 Cross-Language C++ runner
        add_executable(runner_cpp tests/cross_language/runner_cpp.cpp)
        target_include_directories(runner_cpp PRIVATE ${AXMAN_INCLUDES} bindings/cpp/include)
        target_link_libraries(runner_cpp PRIVATE axman_shared)

        # 7. Phase 6.3 Quickstart C++20 Example
        add_executable(quickstart_cpp examples/cpp/quickstart.cpp)
        target_include_directories(quickstart_cpp PRIVATE ${AXMAN_INCLUDES} bindings/cpp/include)
        target_link_libraries(quickstart_cpp PRIVATE axman_shared)
    endif()
endif()

# -----------------------------------------------------------------------------
# Installation Rules
# -----------------------------------------------------------------------------
include(GNUInstallDirs)
install(FILES include/axman_c_api.h include/axman_currying.h include/axman_hash.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/axman)

install(FILES bindings/cpp/include/axman/axman.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/axman)

if(AXMAN_BUILD_SHARED)
    install(TARGETS axman_shared
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(AXMAN_BUILD_STATIC)
    install(TARGETS axman_static
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
