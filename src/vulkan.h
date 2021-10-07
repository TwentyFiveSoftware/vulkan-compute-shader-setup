#pragma once

#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS

#include <vulkan/vulkan.hpp>
#include "structs.h"

struct VulkanBuffer {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
};


class Vulkan {
public:
    explicit Vulkan(Settings settings);

    ~Vulkan();

    void run(Input input);

    Output getOutput();

private:
    const std::vector<const char*> requiredInstanceExtensions = {};
    const std::vector<const char*> requiredDeviceExtensions = {};

    Settings settings;

    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;

    uint32_t computeQueueFamily = 0;
    vk::Queue computeQueue;

    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet descriptorSet;

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;

    vk::CommandPool commandPool;
    vk::CommandBuffer commandBuffer;

    vk::Fence fence;

    VulkanBuffer inputBuffer, outputBuffer;

    void createInstance();

    void pickPhysicalDevice();

    void findQueueFamilies();

    void createLogicalDevice();

    void createCommandPool();

    void createDescriptorSetLayout();

    void createDescriptorPool();

    void createDescriptorSet();

    void createPipelineLayout();

    void createPipeline();

    [[nodiscard]] static std::vector<char> readBinaryFile(const std::string &path);

    void createCommandBuffer();

    void createFence();

    [[nodiscard]] uint32_t findMemoryTypeIndex(const uint32_t &memoryTypeBits,
                                               const vk::MemoryPropertyFlags &properties);

    [[nodiscard]] VulkanBuffer createBuffer(const vk::DeviceSize &size, const vk::Flags<vk::BufferUsageFlagBits> &usage,
                                            const vk::Flags<vk::MemoryPropertyFlagBits> &memoryProperty);

    void destroyBuffer(const VulkanBuffer &buffer) const;

    void executeSingleTimeCommand(const std::function<void(const vk::CommandBuffer &singleTimeCommandBuffer)> &c);

    void createInputOutputBuffers();

    void updateInputBuffer(const Input &input);

};
