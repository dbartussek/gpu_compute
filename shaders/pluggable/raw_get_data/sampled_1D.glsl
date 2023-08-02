#include "../constants.glsl"

layout(set = 0, binding = 0) uniform usampler1D tex;

INPUT_DATA_TYPE get_data_raw(int x, int y, int z) {
    return textureLod(tex, x + (y * TEXTURE_SIZE_X) + (z * TEXTURE_SIZE_X * TEXTURE_SIZE_Y), 0).x;
}
