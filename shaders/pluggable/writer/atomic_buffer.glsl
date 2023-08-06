#include "../constants.glsl"

layout(set = 1, binding = 0, std430) writeonly buffer out_buffer {
    OUTPUT_DATA_TYPE out_value;
};

void main() {
    ivec2 coord = get_coord();

    GetData d = get_data_discarder(coord.x, coord.y);

    OUTPUT_DATA_TYPE expected = get_identity();
    OUTPUT_DATA_TYPE write = d.data;

    while (true) {
        OUTPUT_DATA_TYPE old = atomicCompSwap(out_value, expected, write);

        if (old == expected) {
            break;
        }

        // If we failed, expect to see the returned value and write it + our data
        expected = old;
        write = accumulate(old, d.data);
    }
}
