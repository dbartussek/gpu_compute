#include "../constants.glsl"

#extension GL_EXT_scalar_block_layout: require

layout(set = 0, binding = 0, scalar) readonly buffer b {
    INPUT_DATA_TYPE values[];
};

INPUT_DATA_TYPE get_data_raw(
        int x, int y, int z,
        int size_x, int size_y
        ) {
    int index = x + (y * size_x) + (z * size_x * size_y);
    INPUT_DATA_TYPE data = values[index];
    return data;
}
