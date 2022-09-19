cmake_minimum_required(VERSION 3.22)
project(VulkanComputeShaderSetup)

set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED FATAL_ERROR)

include_directories(PUBLIC ${VULKAN_INCLUDE_DIRS})

add_executable(
        VulkanComputeShaderSetup
        src/main.cpp
        src/vulkan.h
        src/vulkan.cpp
        src/structs.h
)

target_link_libraries(VulkanComputeShaderSetup Vulkan::Vulkan)
