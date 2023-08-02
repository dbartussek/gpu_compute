#include "../constants.glsl"

#extension GL_EXT_scalar_block_layout: require

layout(set = 0, binding = 0, scalar) readonly buffer b {
    INPUT_DATA_TYPE values[];
};

INPUT_DATA_TYPE get_data_raw(int x, int y, int z) {
    int index = x + (y * TEXTURE_SIZE_X) + (z * TEXTURE_SIZE_X * TEXTURE_SIZE_Y);
    INPUT_DATA_TYPE data = values[index];
    return data;
}
