#version 460

#extension GL_KHR_shader_subgroup_quad : require

#include <location.glsl>

layout(location = 0) out uvec2 f_color;

void main() {
    ivec2 coord = get_coord();

    int x = min(subgroupQuadSwapHorizontal(coord.x), coord.x);
    int y = min(subgroupQuadSwapVertical(coord.y), coord.y);

    f_color = ivec2(x, y);
}
