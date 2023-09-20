#include "../constants.glsl"

layout(set = 1, binding = 0, std430) writeonly buffer out_buffer {
    OUTPUT_DATA_TYPE out_value;
};

void main() {
    ivec2 coord = get_coord();

    GetData d = get_data(coord.x, coord.y);
    OUTPUT_DATA_TYPE subgroup_result = subgroupAdd(d.data);

    if (gl_SubgroupInvocationID == 0) {
        atomicAdd(out_value, subgroup_result);
    }
}
