#include "../constants.glsl"

layout(set = 0, binding = 0, r32ui) uniform readonly uimage2D img;

DATA_TYPE get_data_raw(int x, int y, int z) {
    return imageLoad(img, ivec2(x + (y * TEXTURE_SIZE_X), z)).x;
}
