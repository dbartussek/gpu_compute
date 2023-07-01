#include "constants.glsl"

#ifdef COMPUTE_SHADER

layout(local_size_x = 256, local_size_y = 1) in;

layout( push_constant ) uniform pc_l
{
	int z;
} pc;

int get_z() {
    return pc.z;
}

ivec2 get_coord() {
    return ivec2(gl_GlobalInvocationID.xy);
}

#else

layout(location = 0) in vec2 tex_coord;
layout(location = 1) flat in uint instance_id;

int get_z() {
    return int(instance_id);
}

ivec2 get_coord() {
    return ivec2(tex_coord * vec2(TEXTURE_SIZE_X, TEXTURE_SIZE_Y));
}

#endif
