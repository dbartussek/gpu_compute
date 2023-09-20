#include "../constants.glsl"

layout(set = 1, binding = 0, std430) buffer out_buffer {
    OUTPUT_DATA_TYPE out_value;
};

void main() {
    ivec2 coord = get_coord();

    GetData d = get_data(coord.x, coord.y);

    OUTPUT_DATA_TYPE expected = out_value;
    OUTPUT_DATA_TYPE write = d.data;



    if (subgroupElect()) {
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
}
