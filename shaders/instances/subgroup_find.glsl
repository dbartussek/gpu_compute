#version 460

#extension GL_KHR_shader_subgroup_arithmetic : require

#include <location.glsl>

layout(location = 0) out uvec2 f_color;

void main() {
    ivec2 coord = get_coord();

    int x = subgroupMin(coord.x);
    int y = subgroupMin(coord.y);

    f_color = ivec2(x, y);
}
