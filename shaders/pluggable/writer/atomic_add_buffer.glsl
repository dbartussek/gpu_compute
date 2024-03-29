#include "../constants.glsl"

layout(set = 1, binding = 0, std430) writeonly buffer out_buffer {
    OUTPUT_DATA_TYPE out_value;
};

void main() {
    ivec2 coord = get_coord();

    GetData d = get_data(coord.x, coord.y);

    if (!d.do_discard) {
        atomicAdd(out_value, d.data);
    }
}
