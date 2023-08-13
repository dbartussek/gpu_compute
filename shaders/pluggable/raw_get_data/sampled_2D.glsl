#include "../constants.glsl"

layout(set = 0, binding = 0) uniform usampler2D tex;

INPUT_DATA_TYPE get_data_raw(
        int x, int y, int z,
        int size_x, int size_y, int size_z
        ) {
    return textureLod(tex, ivec2(x + (y * size_x), z), 0).x;
}
