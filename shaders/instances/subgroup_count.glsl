#version 460

#extension GL_KHR_shader_subgroup_arithmetic : require

#include <location.glsl>

layout(location = 0) out uvec3 f_color;

void main() {
    ivec2 coord = get_coord();

    uint total_count = gl_SubgroupSize;
    uint active_count = subgroupAdd(gl_HelperInvocation ? 0 : 1);
    uint helper_count = subgroupAdd(gl_HelperInvocation ? 1 : 0);

    f_color = uvec3(total_count, active_count, helper_count);
}
