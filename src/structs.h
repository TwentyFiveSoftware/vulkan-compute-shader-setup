#pragma once

#include <string>

const uint32_t GROUPS_PER_AXIS = 100;

struct Input {
    uint32_t samplesPerGroup;
};

struct Output {
    uint32_t positiveSamples[GROUPS_PER_AXIS * GROUPS_PER_AXIS];
};

struct Settings {
    uint32_t groupCountX, groupCountY, groupCountZ;
    std::string computeShaderFile;
    bool printDebugMessages;
};
