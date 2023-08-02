#include "../constants.glsl"

layout(set = 0, binding = 0, r32ui) uniform readonly uimage1D img;

DATA_TYPE get_data_raw(int x, int y, int z) {
    return imageLoad(img, x + (y * TEXTURE_SIZE_X) + (z * TEXTURE_SIZE_X * TEXTURE_SIZE_Y)).x;
}
