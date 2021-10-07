#include "vulkan.h"
#include <iostream>
#include <set>
#include <fstream>
#include <utility>
#include <chrono>

Vulkan::Vulkan(Settings settings) : settings(std::move(settings)) {
    createInstance();
    pickPhysicalDevice();
    findQueueFamilies();
    createLogicalDevice();
    createCommandPool();
    createInputOutputBuffers();
    createDescriptorSetLayout();
    createDescriptorPool();
    createDescriptorSet();
    createPipelineLayout();
    createPipeline();
    createCommandBuffer();
    createFence();
}

Vulkan::~Vulkan() {
    device.destroyFence(fence);
    device.destroyPipeline(pipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyDescriptorSetLayout(descriptorSetLayout);
    device.destroyDescriptorPool(descriptorPool);
    device.destroyCommandPool(commandPool);

    destroyBuffer(inputBuffer);
    destroyBuffer(outputBuffer);

    device.destroy();
    instance.destroy();
}

void Vulkan::run(Input input) {
    updateInputBuffer(input);

    std::cout << "Starting compute shader execution..." << std::endl;
    auto beginTime = std::chrono::steady_clock::now();

    vk::SubmitInfo submitInfo = {
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer
    };

    computeQueue.submit(1, &submitInfo, fence);

    device.waitForFences(1, &fence, true, UINT64_MAX);
    device.resetFences(fence);

    auto runTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - beginTime).count();

    std::cout << "Compute shader execution took " << runTime << " ms" << std::endl << std::endl;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugMessageFunc(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                                                void* pUserData) {
    std::cout << "["
              << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity)) << " | "
              << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>( messageTypes )) << "]:\n"
              << "id      : " << pCallbackData->pMessageIdName << "\n"
              << "message : " << pCallbackData->pMessage << "\n"
              << std::endl;

    return false;
}

void Vulkan::createInstance() {
    vk::ApplicationInfo applicationInfo = {
            .pApplicationName = "compute-shader-setup",
            .applicationVersion = 1,
            .pEngineName = "compute-shader-setup",
            .engineVersion = 1,
            .apiVersion = VK_API_VERSION_1_2
    };

    std::vector<const char*> enabledLayers =
            {"VK_LAYER_KHRONOS_validation", "VK_LAYER_KHRONOS_synchronization2"};

    if (!settings.printDebugMessages) {
        enabledLayers.clear();
    }

    vk::DebugUtilsMessengerCreateInfoEXT debugMessengerInfo = {
            .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
                               vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                               vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                               vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
            .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                           vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
            .pfnUserCallback = &debugMessageFunc,
            .pUserData = nullptr
    };

    vk::InstanceCreateInfo instanceCreateInfo = {
            .pNext = settings.printDebugMessages ? &debugMessengerInfo : nullptr,
            .pApplicationInfo = &applicationInfo,
            .enabledLayerCount = static_cast<uint32_t>(enabledLayers.size()),
            .ppEnabledLayerNames = enabledLayers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(requiredInstanceExtensions.size()),
            .ppEnabledExtensionNames = requiredInstanceExtensions.data(),
    };

    instance = vk::createInstance(instanceCreateInfo);
}

void Vulkan::pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();

    if (physicalDevices.empty()) {
        throw std::runtime_error("No GPU with Vulkan support found!");
    }

    for (const vk::PhysicalDevice &d: physicalDevices) {
        std::vector<vk::ExtensionProperties> availableExtensions = d.enumerateDeviceExtensionProperties();
        std::set<std::string> requiredExtensions(requiredDeviceExtensions.begin(), requiredDeviceExtensions.end());

        for (const vk::ExtensionProperties &extension: availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        if (requiredExtensions.empty()) {
            physicalDevice = d;
            return;
        }
    }

    throw std::runtime_error("No GPU supporting all required features found!");
}

void Vulkan::findQueueFamilies() {
    std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        bool supportsGraphics = (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics)
                                == vk::QueueFlagBits::eGraphics;
        bool supportsCompute = (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute)
                               == vk::QueueFlagBits::eCompute;

        if (supportsCompute && !supportsGraphics) {
            computeQueueFamily = i;
            break;
        }
    }
}

void Vulkan::createLogicalDevice() {
    float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos = {
            {
                    .queueFamilyIndex = computeQueueFamily,
                    .queueCount = 1,
                    .pQueuePriorities = &queuePriority
            }
    };

    vk::PhysicalDeviceFeatures deviceFeatures = {};

    vk::DeviceCreateInfo deviceCreateInfo = {
            .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos = queueCreateInfos.data(),
            .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtensions.size()),
            .ppEnabledExtensionNames = requiredDeviceExtensions.data(),
            .pEnabledFeatures = &deviceFeatures
    };

    device = physicalDevice.createDevice(deviceCreateInfo);

    computeQueue = device.getQueue(computeQueueFamily, 0);
}

void Vulkan::createCommandPool() {
    commandPool = device.createCommandPool({.queueFamilyIndex = computeQueueFamily});
}

void Vulkan::createDescriptorSetLayout() {
    std::vector<vk::DescriptorSetLayoutBinding> bindings = {
            {
                    .binding = 0,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eCompute
            },
            {
                    .binding = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .descriptorCount = 1,
                    .stageFlags = vk::ShaderStageFlagBits::eCompute
            }
    };

    descriptorSetLayout = device.createDescriptorSetLayout(
            {
                    .bindingCount = static_cast<uint32_t>(bindings.size()),
                    .pBindings = bindings.data()
            });
}

void Vulkan::createDescriptorPool() {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
            {
                    .type = vk::DescriptorType::eUniformBuffer,
                    .descriptorCount = 1
            },
            {
                    .type = vk::DescriptorType::eStorageBuffer,
                    .descriptorCount = 1
            }
    };

    descriptorPool = device.createDescriptorPool(
            {
                    .maxSets = 1,
                    .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                    .pPoolSizes = poolSizes.data()
            });
}

void Vulkan::createDescriptorSet() {
    descriptorSet = device.allocateDescriptorSets(
            {
                    .descriptorPool = descriptorPool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = &descriptorSetLayout
            }).front();


    vk::DescriptorBufferInfo inputBufferInfo = {
            .buffer = inputBuffer.buffer,
            .offset = 0,
            .range = sizeof(Input)
    };

    vk::DescriptorBufferInfo outputBufferInfo = {
            .buffer = outputBuffer.buffer,
            .offset = 0,
            .range = sizeof(Output)
    };

    std::vector<vk::WriteDescriptorSet> descriptorWrites = {
            {
                    .dstSet = descriptorSet,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &inputBufferInfo
            },
            {
                    .dstSet = descriptorSet,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &outputBufferInfo
            }
    };

    device.updateDescriptorSets(static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(),
                                0, nullptr);
}

void Vulkan::createPipelineLayout() {
    pipelineLayout = device.createPipelineLayout(
            {
                    .setLayoutCount = 1,
                    .pSetLayouts = &descriptorSetLayout,
                    .pushConstantRangeCount = 0,
                    .pPushConstantRanges = nullptr
            });
}

void Vulkan::createPipeline() {
    std::vector<char> computeShaderCode = readBinaryFile(settings.computeShaderFile);

    vk::ShaderModuleCreateInfo shaderModuleCreateInfo = {
            .codeSize = computeShaderCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(computeShaderCode.data())
    };

    vk::ShaderModule computeShaderModule = device.createShaderModule(shaderModuleCreateInfo);

    vk::PipelineShaderStageCreateInfo shaderStage = {
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = computeShaderModule,
            .pName = "main",
    };

    vk::ComputePipelineCreateInfo pipelineCreateInfo = {
            .stage = shaderStage,
            .layout = pipelineLayout
    };

    pipeline = device.createComputePipeline(nullptr, pipelineCreateInfo).value;

    device.destroyShaderModule(computeShaderModule);
}

std::vector<char> Vulkan::readBinaryFile(const std::string &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("[Error] Failed to open file at '" + path + "'!");

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

void Vulkan::createCommandBuffer() {
    commandBuffer = device.allocateCommandBuffers(
            {
                    .commandPool = commandPool,
                    .level = vk::CommandBufferLevel::ePrimary,
                    .commandBufferCount = 1
            }).front();

    vk::CommandBufferBeginInfo beginInfo = {};
    commandBuffer.begin(&beginInfo);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

    std::vector<vk::DescriptorSet> descriptorSets = {descriptorSet};
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSets, nullptr);

    commandBuffer.dispatch(settings.groupCountX, settings.groupCountY, settings.groupCountZ);

    commandBuffer.end();
}

void Vulkan::createFence() {
    fence = device.createFence({});
}

uint32_t Vulkan::findMemoryTypeIndex(const uint32_t &memoryTypeBits, const vk::MemoryPropertyFlags &properties) {
    vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((memoryTypeBits & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Unable to find suitable memory type!");
}

VulkanBuffer Vulkan::createBuffer(const vk::DeviceSize &size, const vk::Flags<vk::BufferUsageFlagBits> &usage,
                                  const vk::Flags<vk::MemoryPropertyFlagBits> &memoryProperty) {
    vk::BufferCreateInfo bufferCreateInfo = {
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
    };

    vk::Buffer buffer = device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements memoryRequirements = device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocateInfo = {
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = findMemoryTypeIndex(memoryRequirements.memoryTypeBits, memoryProperty)
    };

    vk::DeviceMemory memory = device.allocateMemory(allocateInfo);

    device.bindBufferMemory(buffer, memory, 0);

    return {
            .buffer = buffer,
            .memory = memory,
    };
}

void Vulkan::destroyBuffer(const VulkanBuffer &buffer) const {
    device.destroyBuffer(buffer.buffer);
    device.freeMemory(buffer.memory);

}

void Vulkan::executeSingleTimeCommand(const std::function<void(const vk::CommandBuffer &singleTimeCommandBuffer)> &c) {
    vk::CommandBuffer singleTimeCommandBuffer = device.allocateCommandBuffers(
            {
                    .commandPool = commandPool,
                    .level = vk::CommandBufferLevel::ePrimary,
                    .commandBufferCount = 1
            }).front();

    vk::CommandBufferBeginInfo beginInfo = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };

    singleTimeCommandBuffer.begin(&beginInfo);

    c(singleTimeCommandBuffer);

    singleTimeCommandBuffer.end();

    vk::SubmitInfo submitInfo = {
            .commandBufferCount = 1,
            .pCommandBuffers = &singleTimeCommandBuffer
    };

    vk::Fence f = device.createFence({});
    computeQueue.submit(1, &submitInfo, f);
    device.waitForFences(1, &f, true, UINT64_MAX);

    device.destroyFence(f);
    device.freeCommandBuffers(commandPool, singleTimeCommandBuffer);
}

void Vulkan::createInputOutputBuffers() {
    inputBuffer = createBuffer(sizeof(Input),
                               vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
                               vk::MemoryPropertyFlagBits::eDeviceLocal);

    outputBuffer = createBuffer(sizeof(Output),
                                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                                vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void Vulkan::updateInputBuffer(const Input &input) {
    VulkanBuffer stagingBuffer = createBuffer(sizeof(Input),
                                              vk::BufferUsageFlagBits::eTransferSrc,
                                              vk::MemoryPropertyFlagBits::eHostVisible |
                                              vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data = device.mapMemory(stagingBuffer.memory, 0, sizeof(Input));
    memcpy(data, &input, sizeof(Input));
    device.unmapMemory(stagingBuffer.memory);

    executeSingleTimeCommand([&](const vk::CommandBuffer &singleTimeCommandBuffer) {
        std::vector<vk::BufferCopy> bufferCopyInfo = {
                {
                        .srcOffset = 0,
                        .dstOffset = 0,
                        .size = sizeof(Input)
                }
        };

        singleTimeCommandBuffer.copyBuffer(stagingBuffer.buffer, inputBuffer.buffer, bufferCopyInfo);
    });

    destroyBuffer(stagingBuffer);
}

Output Vulkan::getOutput() {
    VulkanBuffer hostBuffer = createBuffer(sizeof(Output),
                                           vk::BufferUsageFlagBits::eTransferDst,
                                           vk::MemoryPropertyFlagBits::eHostVisible |
                                           vk::MemoryPropertyFlagBits::eHostCoherent);

    executeSingleTimeCommand([&](const vk::CommandBuffer &singleTimeCommandBuffer) {
        std::vector<vk::BufferCopy> bufferCopyInfo = {
                {
                        .srcOffset = 0,
                        .dstOffset = 0,
                        .size = sizeof(Output)
                }
        };

        singleTimeCommandBuffer.copyBuffer(outputBuffer.buffer, hostBuffer.buffer, bufferCopyInfo);
    });

    Output output;

    void* data = device.mapMemory(hostBuffer.memory, 0, sizeof(Output));
    memcpy(&output, data, sizeof(Output));
    device.unmapMemory(hostBuffer.memory);

    destroyBuffer(hostBuffer);

    return output;
}
