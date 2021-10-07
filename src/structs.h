#pragma once

#include <string>

struct Input {
    uint32_t samplesPerGroup;
    uint32_t groupsPerAxis;
};

struct Output {
    uint32_t positiveSamples;
    uint32_t totalSamples;
};

struct Settings {
    uint32_t groupCountX, groupCountY, groupCountZ;
    std::string computeShaderFile;
    bool printDebugMessages;
};
