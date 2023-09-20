#include "../constants.glsl"

layout(set = 1, binding = 0, std430) writeonly buffer out_buffer {
    OUTPUT_DATA_TYPE out_values[];
};

void main() {
    ivec2 coord = get_coord();
    GetData d = get_data(coord.x, coord.y);


    DATA_TYPE acc = d.data;
    for (uint i = 1; i < gl_SubgroupSize; i*=2) {
    	uint take_from = gl_SubgroupInvocationID + i;
    	DATA_TYPE received = subgroupShuffle(acc, take_from);

    	if (take_from < gl_SubgroupSize) {
    		acc = accumulate(acc, received);
    	}
    }

    if (gl_SubgroupInvocationID == 0) {
        out_values[
            (int(coord.x) + int(coord.y*TEXTURE_SIZE_X)) / gl_SubgroupSize
        ] = acc;
    }
}
