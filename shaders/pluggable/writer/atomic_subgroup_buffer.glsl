#include "../constants.glsl"

layout(set = 1, binding = 0, std430) buffer out_buffer {
    OUTPUT_DATA_TYPE out_value;
};

void main() {
    ivec2 coord = get_coord();

    GetData d = get_data_discarder(coord.x, coord.y);

    DATA_TYPE acc = d.data;
    for (uint i = 1; i < gl_SubgroupSize; i*=2) {
    	uint take_from = gl_SubgroupInvocationID + i;
    	DATA_TYPE received = subgroupShuffle(acc, take_from);

    	if (take_from < gl_SubgroupSize) {
    		acc = accumulate(acc, received);
    	}
    }

    OUTPUT_DATA_TYPE expected = out_value;
    OUTPUT_DATA_TYPE write = acc;

    if (gl_SubgroupInvocationID == 0) {
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
