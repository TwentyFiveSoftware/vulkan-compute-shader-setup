#include <iostream>
#include "vulkan.h"

int main() {
    uint32_t groupsPerAxis = 10;
    uint32_t samplesPerGroup = 100;

    Settings settings = {
            .groupCountX = groupsPerAxis,
            .groupCountY = groupsPerAxis,
            .groupCountZ = 1,
            .computeShaderFile = "shader.comp.spv",
            .printDebugMessages = false
    };

    Input input = {
            .samplesPerGroup = samplesPerGroup,
            .groupsPerAxis = groupsPerAxis
    };


    Vulkan vulkan(settings);

    vulkan.run(input);

    Output output = vulkan.getOutput();

    float pi = 4.0f * (float(output.positiveSamples) / float(output.totalSamples));
    std::cout << "PI approximation: " << pi << std::endl << "(" << output.totalSamples << " samples)" << std::endl;
}
