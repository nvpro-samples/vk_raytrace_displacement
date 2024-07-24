# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22)

project(vk_raytrace_displacement)
set(SAMPLE_EXECUTABLE vk_raytrace_displacement)

set(SOURCE_FILES
  src/main.cpp
  src/raytracing_vk.hpp
  )

# GLSL sources - nvpro_core expects absolute paths
set(GLSL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(GLSL_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/animate_heightmap.comp
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/pathtrace.rchit
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/pathtrace.rgen
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/pathtrace.rmiss
  )
file(GLOB GLSL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.h)

# Look for nvpro_core 1) as a sub-folder 2) at some other locations
if(NOT BASE_DIRECTORY)
  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/external ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../..
    REQUIRED
    DOC "Directory containing nvpro_core"
  )
endif()

# Include nvpro_core makefiles
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

# Set a default CMAKE_INSTALL_PREFIX for nvpro samples
_add_project_definitions(${SAMPLE_EXECUTABLE})

file(RELATIVE_PATH NVPRO_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ${BASE_DIRECTORY}/nvpro_core/)

# Turn on extra nvpro_core features (the same lib can be referenced with
# multiple configurations). Some _add_package_*() nvpro_core specializations
# alter the implementation, in which case samples are requested to compile extra
# source files rather than compile extra static libraries or an all-in-one
# nvpro_core lib. This is done with COMMON_SOURCE_FILES, PACKAGE_SOURCE_FILES,
# PLATFORM_LIBRARIES, LIBRARIES_DEBUG and LIBRARIES_OPTIMIZED.
_add_package_VulkanSDK()
_add_package_ShaderC()
_add_package_ImGUI()
_add_nvpro_core_lib()

add_subdirectory(external/heightmap_rtx)

# Compile shaders to Spir-V, encoded in C header files
compile_glsl(
  SOURCE_FILES ${GLSL_FILES}
  HEADER_FILES ${GLSL_HEADERS}
  DST "${CMAKE_CURRENT_BINARY_DIR}/spirv/generated_spirv"
  VULKAN_TARGET "vulkan1.3"
  HEADER ON
  DEPENDENCY ON
  FLAGS "-I${GLSL_INCLUDE_DIR}" "-I${BASE_DIRECTORY}/nvpro_core" -g
  )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${SAMPLE_EXECUTABLE}
  ${SOURCE_FILES}
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES})

target_include_directories(${SAMPLE_EXECUTABLE} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/spirv)
target_link_libraries(${SAMPLE_EXECUTABLE} nvpro_core heightmap_rtx ${PLATFORM_LIBRARIES})
target_link_libraries(${SAMPLE_EXECUTABLE} debug ${LIBRARIES_DEBUG})
target_link_libraries(${SAMPLE_EXECUTABLE} optimized ${LIBRARIES_OPTIMIZED})

target_compile_definitions(${SAMPLE_EXECUTABLE} PRIVATE
  NVPRO_CORE_DIR="${NVPRO_CORE_DIR}"

  # Compile error from nvpro_core/nvp/perproject_globals.cpp without this,
  # if _add_project_definitions() is commented out
  PROJECT_NAME="${SAMPLE_EXECUTABLE}"
  )

# Defines executable install() targets
_finalize_target(${SAMPLE_EXECUTABLE})

# Add shaders so they show in the VS project
target_sources(${SAMPLE_EXECUTABLE} PRIVATE ${GLSL_FILES} ${GLSL_HEADERS})
source_group("Shaders" FILES ${GLSL_FILES} ${GLSL_HEADERS})

# Debug in the source directory to avoid having a set of relative shader include
# paths just for debugging with visual studio
set_property(TARGET ${SAMPLE_EXECUTABLE} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
