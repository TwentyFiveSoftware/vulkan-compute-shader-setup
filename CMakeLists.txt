cmake_minimum_required(VERSION 3.20)
project(VulkanComputeShaderSetup)

set(CMAKE_CXX_STANDARD 20)

include_directories(
        lib/glm
        lib/Vulkan-Headers/Include
)

# https://vulkan.lunarg.com/
link_directories(
        lib/Vulkan-Headers/Lib
)

add_executable(
        VulkanComputeShaderSetup
        src/main.cpp
        src/vulkan.h
        src/vulkan.cpp
        src/structs.h
)

target_link_libraries(VulkanComputeShaderSetup vulkan-1.lib)
