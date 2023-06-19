#include "../constants.glsl"

layout(set = 0, binding = 0, std430) readonly buffer b {
    uint values[];
};

uint get_data_raw(int x, int y, int z) {
    return values[x + (y * TEXTURE_SIZE_X) + (z * TEXTURE_SIZE_X * TEXTURE_SIZE_Y)];
}
