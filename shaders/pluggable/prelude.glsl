#ifndef PRELUDE_GLSL
#define PRELUDE_GLSL

#ifndef DATA_TYPE
#define DATA_TYPE uint
#endif

#ifndef INPUT_DATA_TYPE
#define INPUT_DATA_TYPE DATA_TYPE
#endif
#ifndef OUTPUT_DATA_TYPE
#define OUTPUT_DATA_TYPE DATA_TYPE
#endif

#include "constants.glsl"


layout(push_constant) uniform pc_l{
    uint data_size;

#ifdef COMPUTE_SHADER
    int z;
#endif
} pc;

bool is_in_bounds(uint index) {
    return index < pc.data_size;
}
bool is_in_bounds(uint x, uint y, uint z) {
    return is_in_bounds(x + (y * TEXTURE_SIZE_X) + (z * TEXTURE_SIZE_X * TEXTURE_SIZE_Y));
}


#include "location.glsl"

#endif
