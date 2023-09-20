#include "../constants.glsl"

layout(set = 1, binding = 0, std430) writeonly buffer out_buffer {
    OUTPUT_DATA_TYPE out_values[];
};

void main() {
    ivec2 coord = get_coord();

    GetData d = get_data(coord.x, coord.y);
    out_values[int(coord.x) + int(coord.y*TEXTURE_SIZE_X)] = d.data;
}
