cmake_minimum_required(VERSION 3.12)
project(OsmAndNativeRender CXX C)

# OsmAnd-core legacy build system expects these two variables
set(CMAKE_TARGET_OS "linux")
set(CMAKE_COMPILER_FAMILY "gcc")

set(OSMAND_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# legacy build system switches
set(CMAKE_SHARED_LIBS_ALLOWED_ON_TARGET TRUE)
set(CMAKE_STATIC_LIBS_ALLOWED_ON_TARGET TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_TARGET_CPU_ARCH_FAMILY "x86")

# platform macro expected by the legacy sources
add_definitions(-DOSMAND_TARGET_OS_linux)

# GCC-15's libstdc++ headers no longer pull in <cstdint> implicitly,
# which breaks Skia's `enum class X : int8_t;` forward declarations
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include cstdint")
endif()

# legacy helper: externals are prepared beforehand via their configure.sh
function(prepare_upstream)
    get_filename_component(_dir "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
    if (NOT EXISTS "${_dir}/upstream.patched")
        message(FATAL_ERROR "${_dir}: upstream.patched missing — run configure.sh first")
    endif()
endfunction()

# legacy helpers: save/restore CMake variables around sub-configures
set(_OSMAND_SAVED_STACK "")
function(save_and_set var value)
    set(_saved "${${var}}")
    set("_OSMAND_SAVED_${var}" "${_saved}" PARENT_SCOPE)
    list(APPEND _OSMAND_SAVED_STACK "${var}")
    set(_OSMAND_SAVED_STACK "${_OSMAND_SAVED_STACK}" PARENT_SCOPE)
    set(${var} "${value}" PARENT_SCOPE)
endfunction()
function(restore var)
    set(${var} "${_OSMAND_SAVED_${var}}" PARENT_SCOPE)
endfunction()

# legacy helper: boost is header-only, nothing to build
function(build_upstream)
endfunction()

# externals + core library + core tools (EyePiece etc.)
include("${OSMAND_ROOT}/core/core.cmake")

# Skia codecs are stubbed unless the corresponding SK_HAS_* macros are set
if (TARGET skia_static)
    set_property(TARGET skia_static APPEND PROPERTY
        COMPILE_DEFINITIONS SK_HAS_LIBPNG SK_HAS_LIBJPEG)
endif()

# official CLI tools: obf-inspector / map-rasterizer (eyepiece) / style-*
include("${OSMAND_ROOT}/tools/cpp-tools/tools.cmake")

# embedded core resources (official OsmAnd-resources bundle)
add_subdirectory("${OSMAND_ROOT}/OsmAnd-resources" "OsmAnd-resources")
set_target_properties(OsmAndCore_ResourcesBundle_shared PROPERTIES
    PREFIX ""
    SUFFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools/cpp-tools/map-rasterizer")
