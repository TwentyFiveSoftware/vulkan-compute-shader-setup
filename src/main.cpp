#include <iostream>
#include <iomanip>
#include "vulkan.h"

int main() {
    Settings settings = {
            .groupCountX = GROUPS_PER_AXIS,
            .groupCountY = GROUPS_PER_AXIS,
            .groupCountZ = 1,
            .computeShaderFile = "shader.comp.spv",
            .printDebugMessages = false
    };

    Input input = {
            .samplesPerGroup = 1000
    };


    std::cout.imbue(std::locale(""));

    Vulkan vulkan(settings);
    vulkan.run(input);
    Output output = vulkan.getOutput();

    uint64_t totalSamples = static_cast<uint64_t>(GROUPS_PER_AXIS) * static_cast<uint64_t>(input.samplesPerGroup) *
                            static_cast<uint64_t>(16);
    totalSamples *= totalSamples;

    uint64_t totalPositiveSamples = 0;

    for (uint32_t positiveSamples: output.positiveSamples) {
        totalPositiveSamples += positiveSamples;
    }

    float pi = 4.0f * (static_cast<float>(totalPositiveSamples) / static_cast<float>(totalSamples));
    std::cout << "PI approximation: " << std::setprecision(16) << pi << std::endl << "(" << totalSamples << " samples)"
              << std::endl;
}
