#include "../constants.glsl"

layout(set = 1, binding = 0, std430) writeonly buffer out_buffer {
    OUTPUT_DATA_TYPE out_values[];
};

void main() {
    ivec2 coord = get_coord();

    // gl_SubgroupSize
    // gl_SubgroupInvocationID

    GetData d = get_data(coord.x, coord.y);
    OUTPUT_DATA_TYPE subgroup_result = subgroupAdd(d.data);

    if (gl_SubgroupInvocationID == 0) {
        out_values[
            (int(coord.x) + int(coord.y*TEXTURE_SIZE_X)) / gl_SubgroupSize
        ] = subgroup_result;
    }
}
